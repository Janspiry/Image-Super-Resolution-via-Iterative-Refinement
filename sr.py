import torch
import json
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-auto_resume', action='store_true')
    
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

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/val_psnr", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None
    
    output_size = opt['datasets']['output_size'] if 'output_size' in opt['datasets'] else 512
    use_3d = bool(opt['datasets']['use_3d'])

    # sampler
    if 'tile_weights' in opt['datasets']['train']:
        with open(opt['datasets']['train']['tile_weights'], 'r') as f:
            tile_weights = {tile_str: weight for tile_str, weight in json.load(f).items()}
        train_sampler = train_data.get_tile_weight_sampler(tile_weights=tile_weights)
    else:
        train_sampler = None

    # dataset
    datatype = opt['datasets']['train']['datatype']
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase, output_size=output_size, use_3d=bool(opt['datasets']['use_3d']))
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase, sampler=train_sampler)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase, output_size=output_size, use_3d=bool(opt['datasets']['use_3d']))
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # Manually check for the most recent checkpoint in this results dir.
    # If auto_resume is not specified, code will load in the resume_state arg if provided,
    # otherwise will not load any weight in (not even the last_opt.th and last_gen.pth).
    if args.auto_resume:
        chkpt_path = opt['path']['checkpoint']
        if not (len(os.listdir(chkpt_path)) == 0 or chpkt_path is None):
            best_weight, best_iters = None, 0
            for chkpt in os.listdir(chkpt_path):
                # Just gonna look for opt weights so we don't repeatedly look.
                if chkpt.endswith('gen.pth'):
                    continue
                # Skipping the last checkpoints.
                if chkpt == 'last_opt.pth':
                    continue
                split = chkpt.split('_')
                iterations = int(split[0][1:])
                epochs = int(split[1][1:])

                if iterations > best_iters:
                    best_weight = chkpt
                    best_iters = iterations
            opt['path']['resume_state'] = os.path.join(opt['path']['checkpoint'], best_weight[:-8])

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    diffusion.set_new_noise_schedule(
    	opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1

            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0

                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                    result_path = 'change_detection_results/'
                    print("result path:", result_path)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)

                        visuals = diffusion.get_current_visuals(datatype=datatype)

                        # LSUN and NAIP reconstruction experiments.
                        if datatype == 'img':
                            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                            Metrics.save_img(
				hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
				sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
				lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
				fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

                        # NAIP reconstruction.
                        elif datatype == 'naip':
                            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            fake_img = Metrics.tensor2img(visuals['Downsampled_NAIP'])  # uint8

                            Metrics.save_img(
                                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                fake_img, '{}/{}_{}_downsampled_naip.png'.format(result_path, current_step, idx))

                        # NAIP generation based on S2 conditioning.
                        elif datatype == 's2' or datatype == 'just-s2':
                            sr_img = Metrics.tensor2img(visuals['SR'])  
                            hr_img = Metrics.tensor2img(visuals['HR'])  
                            s2_img = Metrics.tensor2img(visuals['S2'])

                            if s2_img.shape[0] > 3:
                                s2_img = s2_img[:, :, :3]

                            fake_img = s2_img # placeholder

                            Metrics.save_img(
				hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
				sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

                            if use_3d:
                                # NOTE: unet_3d not properly saving s2_img, shape is (332, 266, 3) ?
                                fake_img = torch.rand((64,64,3))
                            else:
                                fake_img = s2_img
                                Metrics.save_img(s2_img, '{}/{}_{}_s2.png'.format(result_path, current_step, idx))

                        # NAIP generation based on S2 + downsampled NAIP conditioning.
                        elif datatype == 's2_and_downsampled_naip':
                            sr_img = Metrics.tensor2img(visuals['SR'])  
                            hr_img = Metrics.tensor2img(visuals['HR'])  
                            s2_img = Metrics.tensor2img(visuals['S2'])
                            fake_img = s2_img
                            downsampled_naip_img = Metrics.tensor2img(visuals['Downsampled_NAIP'])

                            Metrics.save_img(
				hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
				sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(s2_img, '{}/{}_{}_s2.png'.format(result_path, current_step, idx))
                            Metrics.save_img(downsampled_naip_img, '{}/{}_{}_downsampled_naip.png'.format(result_path, current_step, idx))

                        # generation
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))

                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')

    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)

        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals(datatype=datatype)

            # LSUN and NAIP reconstruction experiments.
            if datatype == 'img':
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # NAIP reconstruction.
            elif datatype == 'naip':
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                fake_img = Metrics.tensor2img(visuals['Downsampled_NAIP'])  # uint8

                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    fake_img, '{}/{}_{}_downsampled_naip.png'.format(result_path, current_step, idx))

            # NAIP generation based on S2 conditioning.
            elif datatype == 's2':
                sr_img = Metrics.tensor2img(visuals['SR'])
                hr_img = Metrics.tensor2img(visuals['HR'])
                s2_img = Metrics.tensor2img(visuals['S2'])

                # Make a gif of all S2 images?
                #import imageio
                #kargs = { 'duration': 0.6 }
                #images = []
                #for i in range(18):
                #    img = s2_img[:, :, i*3:(i*3)+3]
                #    images.append(img)
                #imageio.mimsave(result_path+'/'+str(current_step)+'_'+str(idx)+'_s2-gif.gif', images, **kargs)

                if s2_img.shape[0] > 3:
                    s2_img = s2_img[:, :, :3]

                fake_img = s2_img

                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                Metrics.save_img(s2_img, '{}/{}_{}_s2.png'.format(result_path, current_step, idx))

                # Make a gif?
                #import imageio
                #images = []
                #for img in visuals['SR']:
                #    img = np.transpose(img, (1,2,0))
                #    images.append(img)
                #imageio.mimsave(result_path+'/'+str(current_step)+'_'+str(idx)+'_gif.gif', images)

            # NAIP generation based on S2 + downsampled NAIP conditioning.
            elif datatype == 's2_and_downsampled_naip':
                sr_img = Metrics.tensor2img(visuals['SR'])
                hr_img = Metrics.tensor2img(visuals['HR'])
                s2_img = Metrics.tensor2img(visuals['S2'])
                fake_img = s2_img
                downsampled_naip_img = Metrics.tensor2img(visuals['Downsampled_NAIP'])

                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                Metrics.save_img(s2_img, '{}/{}_{}_s2.png'.format(result_path, current_step, idx))
                Metrics.save_img(downsampled_naip_img, '{}/{}_{}_downsampled_naip.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
