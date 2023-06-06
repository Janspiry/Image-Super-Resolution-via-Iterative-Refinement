import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

        print("Number of netG parameters...", sum(p.numel() for p in self.netG.parameters() if p.requires_grad))

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                print("self.data[SR]:", self.data['SR'].shape)
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False, datatype='img'):
        out_dict = OrderedDict()
        if datatype == 'img':
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        elif datatype == 's2' or datatype == 'just-s2': 
            s2 = self.data['SR']
            out_dict['S2'] = s2.detach().float().cpu()
            out_dict['SR'] = self.SR.detach().float().cpu()  # self.SR is the generated Superresolution
            out_dict['HR'] = self.data['HR'].detach().float().cpu()  # self.data['HR'] is the original ground truth, high-res image
        elif datatype == 's2_and_downsampled_naip':
            s2, downsampled_naip = self.data['SR'][:, :3, :, :], self.data['SR'][:, 3:, :, :]
            out_dict['S2'] = s2.detach().float().cpu()
            out_dict['Downsampled_NAIP'] = downsampled_naip.detach().float().cpu()
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
        elif datatype == 'naip':
            out_dict['Downsampled_NAIP'] = self.data['SR'].detach().float().cpu()
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))

        # Also save to "last_gen.pth" and "last_opt.pth" for Beaker's sake.
        # Will overwrite each checkpoint cycle.
        last_gen_path = os.path.join(self.opt['path']['checkpoint'], 'last_gen.pth')
        last_opt_path = os.path.join(self.opt['path']['checkpoint'], 'last_opt.pth')

        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        torch.save(state_dict, last_gen_path)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
        torch.save(opt_state, last_opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        print("LOADING NETWORK...")

        # For resuming state as the original code does:
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            print("loading in:", gen_path, " and ", opt_path) 
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
            return

        # For resuming the last gen and opt states; specifically for Beaker scenario.
        load_gen_path = self.opt['path']['resume_gen_state']
        load_opt_path = self.opt['path']['resume_opt_state']
        if load_gen_path is not None and load_opt_path is not None:

            if os.path.exists(load_gen_path):
                logger.info(
                    'Loading pretrained model for G [{:s}] ...'.format(load_gen_path))

                # gen
                network = self.netG
                if isinstance(self.netG, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(
                    load_gen_path), strict=(not self.opt['model']['finetune_norm']))

            if os.path.exists(load_opt_path):
                if self.opt['phase'] == 'train':
                    # optimizer
                    opt = torch.load(load_opt_path)
                    self.optG.load_state_dict(opt['optimizer'])
                    self.begin_step = opt['iter']
                    self.begin_epoch = opt['epoch']

