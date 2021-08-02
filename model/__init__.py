import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']['which_model_G']

    if model == 'DDPM':
        from .DDPM import DDPM as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
