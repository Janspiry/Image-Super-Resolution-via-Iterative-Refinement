import torch
import skimage.io
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

from metrics import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
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

    # If not resuming from last checkpoint and just trying to load in weights, it should default to here.
    current_epoch = 2000
    current_step = 1000
    if (opt['path']['resume_gen_state'] and opt['path']['resume_opt_state']) or opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase, output_size=opt['datasets']['output_size'])
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    save_dir = '/data/piperw/data/mturk/worldstrat/' # temporary save dir for cvpr
    avg_psnr, avg_ssim = [], []

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        #hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        #fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        # bias adjustment using ground truth
        output = visuals['SR'][:, -1, :, :, :]

        save_path = save_dir + str(idx-1) + '/sr3.png'
        output = Metrics.tensor2img(output)
        skimage.io.imsave(save_path, output)
        continue

        """
        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sr_img = Metrics.tensor2img(sr_img[-1, :, :, :])
            Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            # If you want to save each individual in the sampling process...
            #sample_num = sr_img.shape[0]
            #for iter in range(0, sample_num):
            #    Metrics.save_img(
            #        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # If you want to save each individual in the sampling process, as a grid...
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            print(
            #Metrics.save_img(
            #    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            # changed the indexing here from [-1] to [:, -1, :, :, :]
            Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][:, -1, :, :, :]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
        """

        sr_img = Metrics.tensor2img(visuals['SR'][:, -1, :, :, :])
        Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        #Metrics.save_img(
        #    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        #Metrics.save_img(
        #        fake_img[:,:,:3], '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        eval_psnr = calculate_psnr(hr_img, sr_img, 0)
        eval_ssim = calculate_ssim(hr_img, sr_img, 0)
        print(eval_psnr, eval_ssim)

        avg_psnr.append(eval_psnr)
        avg_ssim.append(eval_ssim)

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    avg_psnr = sum(avg_psnr) / len(avg_psnr)
    avg_ssim = sum(avg_ssim) / len(avg_ssim)

    print("Avg PSNR:", avg_psnr)
    print("Avg SSIM:", avg_ssim)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
