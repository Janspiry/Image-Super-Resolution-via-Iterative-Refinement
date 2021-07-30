
import torch
from torch import optim
import data as Data
import model as Model
import argparse
import logging
import logger as Logger
import metrics
from tensorboardX import SummaryWriter
import copy
import os


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def set_device(x):
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None:
                x[key] = item.cuda()
    elif isinstance(x, list):
        for item in x:
            if item is not None:
                item = item.cuda()
    else:
        x = x.cuda()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/basic_sr.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either training or generation', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt['model'])
    ema = copy.deepcopy(diffusion)
    if opt['train']["optimizer"]["type"] == 'adam':
        optimizer = optim.Adam(diffusion.parameters(),
                               lr=opt['train']["optimizer"]["lr"])

    logger.info('Initial Model Finished')

    # Train
    current_step = 0
    current_epoch = 0
    n_iter = opt['train']['n_iter']
    step_start_ema = opt['train']["ema_scheduler"]['step_start_ema']
    update_ema_every = opt['train']["ema_scheduler"]['update_ema_every']
    ema_decay = opt['train']["ema_scheduler"]['ema_decay']

    diffusion = set_device(diffusion)
    ema = set_device(ema)

    if opt['phase'] == 'train':
        while True:
            for _, train_data in enumerate(train_loader):
                train_data = set_device(train_data)
                # training
                optimizer.zero_grad()

                loss = diffusion(train_data)
                loss.backward()
                optimizer.step()

                if current_step % update_ema_every == 0:
                    accumulate(
                        ema, diffusion, 0 if current_step < step_start_ema else ema_decay
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

                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                    for _,  val_data in enumerate(val_loader):
                        val_data = set_device(val_data)
                        idx += 1
                        sr_img = val_data['LR']
                        gt_img = val_data['HR']
                        sr_img = diffusion.super_resolution(sr_img)

                        gt_img = metrics.tensor2img(gt_img)
                        sr_img = metrics.tensor2img(sr_img)
                        # generation
                        avg_psnr += metrics.calculate_psnr(
                            (1+sr_img) * 127.5, (1+gt_img) * 127.5)

                        metrics.save_img(
                            gt_img, '{}/{}_real.png'.format(result_path, idx))
                        metrics.save_img(
                            sr_img, '{}/{}_gen.png'.format(result_path, idx))
                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    current_step += 1
                    if current_step > n_iter:
                        break
            if current_step > n_iter:
                break
            current_epoch += 1
        # save model
        logger.info('Saving the final model.')

        logger.info('End of training.')
    elif opt['phase'] == 'val':
        avg_psnr = 0.0
        idx = 0
        result_path = '{}/{}'.format(opt['path']
                                     ['results'], 'evaluation')
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            val_data = set_device(val_data)
            idx += 1
            sr_img = val_data['LR']
            gt_img = val_data['HR']
            sr_img = diffusion.super_resolution(sr_img)

            gt_img = metrics.tensor2img(gt_img)
            sr_img = metrics.tensor2img(sr_img)
            # generation
            avg_psnr += metrics.calculate_psnr(
                (1+sr_img) * 127.5, (1+gt_img) * 127.5)

            metrics.save_img(
                gt_img, '{}/{}_real.png'.format(result_path, idx))
            metrics.save_img(
                sr_img, '{}/{}_gen.png'.format(result_path, idx))
        avg_psnr = avg_psnr / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
            current_epoch, current_step, avg_psnr))
        # tensorboard logger
        tb_logger.add_scalar('psnr', avg_psnr, current_step)
