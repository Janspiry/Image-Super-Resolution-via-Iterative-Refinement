from model import diffusion
import torch
from torch import optim
import data
import model
import argparse
import logging
import logger
import metrics
from tensorboardX import SummaryWriter
import copy


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/basic_sr.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either training or generation')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    # parse configs
    opt = logger.parse(parser.parse_args().opt, is_train=True)
    # Convert to NoneDict, which return None for missing key.
    opt = logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = data.create_dataset(dataset_opt)
            train_loader = data.create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = data.create_dataset(dataset_opt)
            val_loader = data.create_dataloader(val_set, dataset_opt)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = model.create_model(opt['model'])
    ema = copy.deepcopy(diffusion)
    if opt['training']["optimizer"]["type"] == 'adam':
        opt = optim.Adam(diffusion.parameters(),
                         lr=opt['training']["optimizer"]["lr"])

    logger.info('Initial Model Finished')

    # Train
    current_step = 0
    current_epoch = 0
    n_iter = opt['training']["scheduler"]
    step_start_ema = opt['training']["scheduler"]['step_start_ema'],
    update_ema_every = opt['training']["scheduler"]['update_ema_every'],
    ema_decay = opt['training']["scheduler"]['ema_decay'],
    while True:
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break

            # training
            opt.zero_grad()

            loss = diffusion(train_data)
            loss.backward()
            opt.step()

            if current_step % update_ema_every == 0:
                accumulate(
                    ema, model.module, 0 if current_step < step_start_ema else ema_decay
                )
            # log
            if current_step % opt['train']['print_freq'] == 0:
                message = '<epoch:{:3d}, iter:{:8,d}, loss:{:.3e}> '.format(
                    current_epoch, current_step, loss)
                tb_logger.add_scalar("loss", loss, current_step)
                logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    sr_img = val_data['LR']
                    gt_img = val_data['HR']

                    # generation
                    avg_psnr += metrics.calculate_psnr(
                        (1+sr_img) * 127.5, (1+gt_img) * 127.5)

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    current_epoch, current_step, avg_psnr))
                # tensorboard logger
                tb_logger.add_scalar('psnr', avg_psnr, current_step)

        if current_step > n_iter:
            break
        current_epoch += 1

    # save model
    logger.info('Saving the final model.')

    logger.info('End of training.')
