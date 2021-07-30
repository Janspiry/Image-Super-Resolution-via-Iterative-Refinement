from .diffusion import GaussianDiffusion, make_beta_schedule
from .unet import UNet
import logging
from torch import nn
logger = logging.getLogger('base')


def get_network_description(network):
    '''Get the string and total parameters of the network'''
    if isinstance(network, nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n


def print_network(net):
    s, n = get_network_description(net)
    if isinstance(net, nn.DataParallel):
        net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                         net.module.__class__.__name__)
    else:
        net_struc_str = '{}'.format(net.__class__.__name__)
    logger.info('Network G structure: {}, with parameters: {:,d}'.format(
        net_struc_str, n))
    logger.info(s)


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
        attn_res=opt['unet']['attn_res'],
        res_blocks=opt['unet']['res_blocks'],
        dropout=opt['unet']['dropout'],
        image_size=opt['diffusion']['image_size']
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
    logger.info('Model [{:s}] is created.'.format('DPPM'))
    print_network(model)
    return diffusion
