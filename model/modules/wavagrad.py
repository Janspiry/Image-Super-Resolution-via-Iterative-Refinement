import numpy as np

import torch

from model.base import BaseModule
from model.nn import WaveGradNN


class WaveGrad(BaseModule):
    """
    WaveGrad diffusion process as described in WaveGrad paper
    (link: https://arxiv.org/pdf/2009.00713.pdf).
    Implementation adopted from `Denoising Diffusion Probabilistic Models`
    repository (link: https://github.com/hojonathanho/diffusion,
    paper: https://arxiv.org/pdf/2006.11239.pdf).
    """

    def __init__(self, config):
        super(WaveGrad, self).__init__()
        # Setup noise schedule
        self.noise_schedule_is_set = False

        # Backbone neural network to model noise
        self.total_factor = np.product(config.model_config.factors)
        assert self.total_factor == config.data_config.hop_length, \
            """Total factor-product should be equal to the hop length of STFT."""
        self.nn = WaveGradNN(config)

    def set_new_noise_schedule(
        self,
        init=torch.linspace,
        init_kwargs={'steps': 50, 'start': 1e-6, 'end': 1e-2}
    ):
        """
        Sets sampling noise schedule. Authors in the paper showed
        that WaveGrad supports variable noise schedules during inference.
        Thanks to the continuous noise level conditioning.
        :param init (callable function, optional): function which initializes betas
        :param init_kwargs (dict, optional): dict of arguments to be pushed to `init` function.
            Should always contain the key `steps` corresponding to the number of iterations to be done by the model.
            This is done so because `torch.linspace` has this argument named as `steps`.
        """
        assert 'steps' in list(init_kwargs.keys()), \
            '`init_kwargs` should always contain the key `steps` corresponding to the number of iterations to be done by the model.'
        n_iter = init_kwargs['steps']

        betas = init(**init_kwargs)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.FloatTensor([1]), alphas_cumprod[:-1]])
        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1]), alphas_cumprod])
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for posterior q(y_n|y_0)
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        # For WaveGrad special continuous noise level conditioning
        self.sqrt_alphas_cumprod_prev = alphas_cumprod_prev_with_last.sqrt().numpy()
        sqrt_recip_alphas_cumprod = (1 / alphas_cumprod).sqrt()
        sqrt_alphas_cumprod_m1 = (
            1 - alphas_cumprod).sqrt() * sqrt_recip_alphas_cumprod
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod_m1', sqrt_alphas_cumprod_m1)

        # Calculations for posterior q(y_{t-1} | y_t, y_0)
        posterior_variance = betas * \
            (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_variance = torch.stack(
            [posterior_variance, torch.FloatTensor([1e-20] * n_iter)])
        posterior_log_variance_clipped = posterior_variance.max(
            dim=0).values.log()
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * \
            alphas.sqrt() / (1 - alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped',
                             posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

        self.n_iter = n_iter
        self.noise_schedule_kwargs = {'init': init, 'init_kwargs': init_kwargs}
        self.noise_schedule_is_set = True

    def sample_continuous_noise_level(self, batch_size, device):
        """
        Samples continuous noise level sqrt(alpha_cumprod).
        This is what makes WaveGrad different from other Denoising Diffusion Probabilistic Models.
        """
        s = np.random.choice(range(1, self.n_iter + 1), size=batch_size)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[s-1],
                self.sqrt_alphas_cumprod_prev[s],
                size=batch_size
            )
        ).to(device)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)

    def q_sample(self, y_0, continuous_sqrt_alpha_cumprod=None, eps=None):
        """
        Efficiently computes diffusion version y_t from y_0 using a closed form expression:
            y_t = sqrt(alpha_cumprod)_t * y_0 + sqrt(1 - alpha_cumprod_t) * eps,
            where eps is sampled from a standard Gaussian.
        """
        batch_size = y_0.shape[0]
        continuous_sqrt_alpha_cumprod \
            = self.sample_continuous_noise_level(batch_size, device=y_0.device) \
            if isinstance(eps, type(None)) else continuous_sqrt_alpha_cumprod
        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0)
        # Closed form signal diffusion
        outputs = continuous_sqrt_alpha_cumprod * y_0 + \
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * eps
        return outputs

    def q_posterior(self, y_start, y, t):
        """
        Computes reverse (denoising) process posterior q(y_{t-1}|y_0, y_t, x)
        parameters: mean and variance.
        """
        posterior_mean = self.posterior_mean_coef1[t] * \
            y_start + self.posterior_mean_coef2[t] * y
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def predict_start_from_noise(self, y, t, eps):
        """
        Computes y_0 from given y_t and reconstructed noise.
        Is needed to reconstruct the reverse (denoising)
        process posterior q(y_{t-1}|y_0, y_t, x).
        """
        return self.sqrt_recip_alphas_cumprod[t] * y - self.sqrt_alphas_cumprod_m1[t] * eps

    def p_mean_variance(self, mels, y, t, clip_denoised: bool):
        """
        Computes Gaussian transitions of Markov chain at step t
        for further computation of y_{t-1} given current state y_t and features.
        """
        batch_size = mels.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(mels)
        eps_recon = self.nn(mels, y, noise_level)
        y_recon = self.predict_start_from_noise(y, t, eps_recon)

        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            y_start=y_recon, y=y, t=t)
        return model_mean, posterior_log_variance

    def compute_inverse_dynamics(self, mels, y, t, clip_denoised=True):
        """
        Computes reverse (denoising) process dynamics. Closely related to the idea of Langevin dynamics.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param y (torch.Tensor): previous state from dynamics trajectory
        :param clip_denoised (bool, optional): clip signal to [-1, 1]
        :return (torch.Tensor): next state
        """
        model_mean, model_log_variance = self.p_mean_variance(
            mels, y, t, clip_denoised)
        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)
        return model_mean + eps * (0.5 * model_log_variance).exp()

    def sample(self, mels, store_intermediate_states=False):
        """
        Samples speech waveform via progressive denoising of white noise with guidance of mels-epctrogram.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional): whether to store dynamics trajectory or not
        :return ys (list of torch.Tensor) (if store_intermediate_states=True)
            or y_0 (torch.Tensor): predicted signals on every dynamics iteration of shape [B, T]
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            batch_size, T = mels.shape[0], mels.shape[-1]
            ys = [torch.randn(batch_size, T*self.total_factor,
                              dtype=torch.float32).to(device)]
            t = self.n_iter - 1
            while t >= 0:
                y_t = self.compute_inverse_dynamics(mels, y=ys[-1], t=t)
                ys.append(y_t)
                t -= 1
            return ys if store_intermediate_states else ys[-1]

    def compute_loss(self, mels, y_0):
        """
        Computes loss between GT Gaussian noise and reconstructed noise by model from diffusion process.
        :param mels (torch.Tensor): mel-spectrograms acoustic features of shape [B, n_mels, T//hop_length]
        :param y_0 (torch.Tensor): GT speech signals
        :return loss (torch.Tensor): loss of diffusion model
        """
        self._verify_noise_schedule_existence()

        # Sample continuous noise level
        batch_size = y_0.shape[0]
        continuous_sqrt_alpha_cumprod \
            = self.sample_continuous_noise_level(batch_size, device=y_0.device)
        eps = torch.randn_like(y_0)

        # Diffuse the signal
        y_noisy = self.q_sample(y_0, continuous_sqrt_alpha_cumprod, eps)

        # Reconstruct the added noise
        eps_recon = self.nn(mels, y_noisy, continuous_sqrt_alpha_cumprod)
        loss = torch.nn.L1Loss()(eps_recon, eps)
        return loss

    def forward(self, mels, store_intermediate_states=False):
        """
        Generates speech from given mel-spectrogram.
        :param mels (torch.Tensor): mel-spectrogram tensor of shape [1, n_mels, T//hop_length]
        :param store_intermediate_states (bool, optional):
            flag to set return tensor to be a set of all states of denoising process 
        """
        self._verify_noise_schedule_existence()

        return self.sample(
            mels, store_intermediate_states
        )

    def _verify_noise_schedule_existence(self):
        if not self.noise_schedule_is_set:
            raise RuntimeError(
                'No noise schedule is found. Specify your noise schedule '
                'by pushing arguments into `set_new_noise_schedule(...)` method. '
                'For example: '
                "`wavegrad.set_new_noise_level(init=torch.linspace, init_kwargs=\{'steps': 50, 'start': 1e-6, 'end': 1e-2\})`."
            )
