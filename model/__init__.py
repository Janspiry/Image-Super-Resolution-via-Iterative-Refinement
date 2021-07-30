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
        in_channel=opt['unet']['in_channel'],
        out_channel=opt['unet']['out_channel'],
        inner_channel=opt['unet']['inner_channel'],
        channel_mults=opt['unet']['channel_multiplier'],
        attn_mults=opt['unet']['attn_mults'],
        res_blocks=opt['unet']['res_blocks'],
        dropout=opt['unet']['dropout']
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=opt['diffusion']['image_size'],
        channels=opt['diffusion']['channels'],
        timesteps=opt['beta_schedule']['n_timestep'],
        loss_type='l1',    # L1 or L2
        betas=beta,
        conditional=opt['diffusion']['conditional']
    )
    logger.info('Model [{:s}] is created.')
    return diffusion
