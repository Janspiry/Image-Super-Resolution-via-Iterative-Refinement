from .diffusion import GaussianDiffusion, make_beta_schedule
from .unet import UNet
import logging
logger = logging.getLogger('base')


def create_model(opt):
    beta = make_beta_schedule(
        schedule=opt['beta_schedule']['schedule'],
        n_timestep=opt['beta_schedule']['n_timestep'],
        linear_start=opt['beta_schedule']['linear_start'],
        linear_end=opt['beta_schedule']['linear_end'])
    model = UNet(
        dim=opt['unet']['channel'],
        dim_mults=opt['unet']['channel_multiplier']
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=opt['diffusion']['image_size'],
        channels=opt['diffusion']['channels'],
        timesteps=opt['beta_schedule']['n_timestep'],
        loss_type='l1',    # L1 or L2
        betas=beta
    )
    logger.info('Model [{:s}] is created.')
    return diffusion
