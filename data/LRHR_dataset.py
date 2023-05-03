from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import skimage.io
import os
import cv2
import json
import glob
import numpy as np
import torch
from torchvision.transforms import functional as trans_fn


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False,
                    n_s2_images=-1, downsample_res=-1):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.n_s2_images = n_s2_images
        self.downsample_res = downsample_res

        # Conditioning on S2.
        if datatype == 's2' or datatype == 's2_and_downsampled_naip':
            self.s2_path = '/data/first_ten_million/s2/'
            self.naip_path = '/data/first_ten_million/naip/'

            # Open the metadata file that contains naip_chip:s2_tiles mappings.
            meta_file = open('/data/first_ten_million/metadata/naip_to_s2.json')
            self.meta = json.load(meta_file)
            self.naip_chips = list(self.meta.keys())

            # Using the metadata, create list of [naip_path, [s2_paths]] sets.
            self.datapoints = []
            for k,v in self.meta.items():
                naip_name, naip_chip = k[:-12], k[-11:]
                naip_path = os.path.join(self.naip_path, naip_name, 'tci', naip_chip+'.png')

                s2_list = [os.path.join(self.s2_path, x[0], 'tci', x[1]+'.png') for x in v]
                s2_paths = random.sample(s2_list, self.n_s2_images)

                self.datapoints.append([naip_path, s2_paths])

            self.data_len = len(self.datapoints)
            print("number of naip chips:", self.data_len, " & len(meta):", len(self.meta))
        
        # NAIP reconstruction, build downsampled version on-the-fly.
        elif datatype == 'naip':
            self.naip_path = '/data/first_ten_million/naip/'

            # Open the metadata file that contains naip_chip:s2_tiles mappings.
            meta_file = open('/data/first_ten_million/metadata/naip_to_s2.json')
            self.meta = json.load(meta_file)
            self.naip_chips = list(self.meta.keys())

            # Build list of NAIP chip paths.
            self.datapoints = []
            for k,v in self.meta.items():
                naip_name, naip_chip = k[:-12], k[-11:]
                naip_path = os.path.join(self.naip_path, naip_name, 'tci', naip_chip+'.png')

                self.datapoints.append(naip_path)
            self.data_len = len(self.datapoints)

        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        # Conditioning on S2, or S2 and downsampled NAIP.
        if self.datatype == 's2' or self.datatype == 's2_and_downsampled_naip':
            datapoint = self.datapoints[index]
            naip_path, s2_paths = datapoint[0], datapoint[1]

            # Load the 512x512 NAIP chip.
            naip_chip = skimage.io.imread(naip_path)

	    # Extract components from the NAIP chip filepath.
            split = naip_path.split('/')
            chip = split[-1][:-4]
            tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16  # s2 tile that contains the naip chip

            # For each S2 tile, we want to read in the image then find and extract the 32x32 chunk corresponding to NAIP chip.
            s2_chunks = []
            for s2_path in s2_paths:
                s2_img = skimage.io.imread(s2_path)
                
                s2_left_corner = tile[0] * 16, tile[1] * 16
                diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]
                s2_chunk = s2_img[diffs[1]*32 : (diffs[1]+1)*32, diffs[0]*32 : (diffs[0]+1)*32, :]

                s2_chunk = torch.permute(torch.from_numpy(s2_chunk), (2, 0, 1))
                s2_chunk = trans_fn.resize(s2_chunk, 512, Image.BICUBIC)
                s2_chunk = trans_fn.center_crop(s2_chunk, 512)
                s2_chunk = torch.permute(s2_chunk, (1, 2, 0)).numpy()
                s2_chunks.append(s2_chunk)

            # If conditioning on downsampled naip (along with S2), need to downsample original NAIP datapoint and upsample
            # it to get it to the size of the other inputs.
            if self.datatype == 's2_and_downsampled_naip':
                downsampled_naip = cv2.resize(naip_chip, dsize=(self.downsample_res,self.downsample_res), interpolation=cv2.INTER_CUBIC)
                downsampled_naip = cv2.resize(downsampled_naip, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

                if len(s2_chunks) == 1:
                    s2_chunk = s2_chunks[0]

                    [s2_chunk, downsampled_naip, naip_chip] = Util.transform_augment(
                                                   [s2_chunk, downsampled_naip, naip_chip], split=self.split, min_max=(-1, 1))
                    img_SR = torch.cat((s2_chunk, downsampled_naip))
                    img_HR = naip_chip
                else:
                    print("TO BE IMPLEMENTED")
                    [s2_chunks, downsampled_naip, naip_chip] = Util.transform_augment(
                                                    [s2_chunks, downsampled_naip, naip_chip], split=self.split, min_max=(-1, 1), multi_s2=True)
                    img_SR = torch.cat((torch.stack(s2_chunks), downsampled_naip))
                    img_HR = naip_chip

            elif self.datatype == 's2':

                if len(s2_chunks) == 1:
                    s2_chunk = s2_chunks[0]

                    [img_SR, img_HR] = Util.transform_augment(
				    [s2_chunk, naip_chip], split=self.split, min_max=(-1, 1))
                else:
                    [s2_chunks, img_HR] = Util.transform_augment(
                                    [s2_chunks, naip_chip], split=self.split, min_max=(-1, 1), multi_s2=True)

                    img_SR = torch.cat(s2_chunks)

            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

        elif self.datatype == 'naip':
            naip_path = self.datapoints[index]

            # Load the 512x512 NAIP chip.
            naip_chip = skimage.io.imread(naip_path)

            # Create the downsampled version on-the-fly.
            downsampled_naip = cv2.resize(naip_chip, dsize=(self.downsample_res,self.downsample_res), interpolation=cv2.INTER_CUBIC)
            downsampled_naip = cv2.resize(downsampled_naip, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

            [img_SR, img_HR] = Util.transform_augment([downsampled_naip, naip_chip], split=self.split, min_max=(-1, 1))

            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")

        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
