import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

import data.util as Util
from metrics import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_txt', type=str, help="Path to a txt file with a list of naip filepaths.")
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
    #for phase, dataset_opt in opt['datasets'].items():
    #    if phase == 'val':
    #        val_set = Data.create_dataset(dataset_opt, phase, output_size=opt['datasets']['output_size'])
    #        val_loader = Data.create_dataloader(
    #            val_set, dataset_opt, phase)
    #logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    avg_psnr, avg_ssim = [], []

    txt = open(args.data_txt)
    fps = txt.readlines()
    for idx,png in enumerate(fps):
        print("Processing....", idx)

        png = png.replace('\n', '')

        # Want to save the super-resolved imagery in the same filepath structure 
        # as the Sentinel-2 imagery, but in a different directory specified by args.save_path
        # for easy comparison.
        file_info = png.split('/')
        chip = file_info[-1][:-4]
        save_dir = os.path.join(save_path, chip)
        os.makedirs(save_dir, exist_ok=True)

        # Load and format NAIP image as diffusion code expects.
        naip_chip = skimage.io.imread(png)
        #skimage.io.imsave(save_dir + '/naip.png', naip_im)

        chip = chip.split('_')
        tile = int(chip[0]) // 16, int(chip[1]) // 16

        s2_left_corner = tile[0] * 16, tile[1] * 16
        diffs = int(chip[0]) - s2_left_corner[0], int(chip[1]) - s2_left_corner[1]

        # Load and format S2 images as diffusion code expects.
        s2_path = '/data/piperw/data/full_dataset/s2_condensed/' + str(tile[0])+'_'+str(tile[1]) + '/' + str(diffs[1])+'_'+str(diffs[0]) + '.png'
        s2_images = skimage.io.imread(s2_path)
        s2_chunks = np.reshape(s2_images, (-1, 32, 32, 3))

        goods, bads = [], []
        for i,ts in enumerate(s2_chunks):
            if [0, 0, 0] in ts:
                bads.append(i)
            else:
        if len(goods) >= n_s2_images:
            rand_indices = random.sample(goods, n_s2_images)
        else:
            need = n_s2_images - len(goods)
            rand_indices = goods + random.sample(bads, need)

        s2_chunks = [s2_chunks[i] for i in rand_indices]
        s2_chunks = np.array(s2_chunks)
        up_s2_chunk = torch.permute(torch.from_numpy(s2_chunks), (0, 3, 1, 2))
        up_s2_chunk = trans_fn.resize(up_s2_chunk, self.output_size, Image.BICUBIC, antialias=True)
        s2_chunks = torch.permute(up_s2_chunk, (0, 2, 3, 1)).numpy()
        [s2_chunks, img_HR] = Util.transform_augment(
                        [s2_chunks, naip_chip], split=self.split, min_max=(-1, 1), multi_s2=True)
        img_SR = torch.cat(s2_chunks)

        val_data = {'HR': img_HR, 'SR': img_SR, 'Index': idx}

        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

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
        Metrics.save_img(sr_img, save_dir + '/sr3.png')

        #Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
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
