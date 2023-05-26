from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import skimage.io
import os
import random
import cv2
import json
import glob
import numpy as np
import torch
from torchvision.transforms import functional as trans_fn
import glob


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False,
                    n_s2_images=-1, downsample_res=-1, output_size=512, max_tiles=-1, specify_val=True):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.n_s2_images = n_s2_images
        self.downsample_res = downsample_res
        self.output_size = output_size
        self.max_tiles = max_tiles

        print("SELF.DATATYPE:", self.datatype)

        # Paths to the imagery.
        self.s2_path = os.path.join(dataroot, 's2_condensed')
        if self.output_size == 512:
            self.naip_path = os.path.join(dataroot, 'naip')
        elif self.output_size == 256:
            self.naip_path = os.path.join(dataroot, 'naip_256')
        elif self.output_size == 128:
            self.naip_path = os.path.join(dataroot, 'naip_128')
        elif self.output_size == 64:
            self.naip_path = os.path.join(dataroot, 'naip_64')
        elif self.output_size == 32:
            self.naip_path = os.path.join(dataroot, 'naip')
        else:
            print("WARNING: output size not supported yet.")

        # Load in the list of naip images that we want to use for val.
        specify_val = True
        self.val_fps = []
        if specify_val:
            val_fps_f = open('held_out.txt')
            val_fps = val_fps_f.readlines()
            for fp in val_fps:
                fp = fp[:-1]
                self.val_fps.append(os.path.join(self.naip_path, fp))

        self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)
        print("self.naip chips:", len(self.naip_chips))

        # Conditioning on S2.
        if datatype == 's2' or datatype == 's2_and_downsampled_naip' or datatype == 'just-s2':

            self.datapoints = []
            for n in self.naip_chips:

		# If this is the train dataset, ignore the subset of images that we want to use for validation.
                if self.split == 'train' and specify_val and (n in self.val_fps):
                    print("split == train and n in val_fps, skipping....")
                    continue
		# If this is the validation dataset, ignore any images that aren't in the subset.
                if self.split == 'val' and specify_val and not (n in self.val_fps):
                    continue

                # ex. /data/first_ten_million/naip/m_2808033_sw_17_060_20191202/tci/36046_54754.png
                split_path = n.split('/')
                chip = split_path[-1][:-4]
                tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16
                s2_left_corner = tile[0] * 16, tile[1] * 16
                diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]

                s2_path = os.path.join(self.s2_path, str(tile[0])+'_'+str(tile[1]), str(diffs[1])+'_'+str(diffs[0])+'.png')

                self.datapoints.append([n, s2_path])

                # Only add 'max_tiles' datapoints to the datapoints list if specified.
                if self.max_tiles != -1 and len(self.datapoints) >= self.max_tiles:
                    break

            self.data_len = len(self.datapoints)

        # NAIP reconstruction, build downsampled version on-the-fly.
        elif datatype == 'naip':

            # Build list of NAIP chip paths.
            self.datapoints = []
            for n in self.naip_chips:

                # If this is the train dataset, ignore the subset of images that we want to use for validation.
                if self.split == 'train' and self.specify_val and (naip_path in self.val_fps):
                    continue
                # If this is the validation dataset, ignore any images that aren't in the subset.
                if self.split == 'val' and self.specify_val and not (naip_path in self.val_fps):
                    continue

                self.datapoints.append(n)
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

        print("Data length:", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        # Conditioning on S2, or S2 and downsampled NAIP.
        if self.datatype == 's2' or self.datatype == 's2_and_downsampled_naip' or self.datatype == 'just-s2':

            hack = 0
            while(True):
                try:
                    datapoint = self.datapoints[index]
                    naip_path, s2_path = datapoint[0], datapoint[1]

                    # Load the 512x512 NAIP chip.
                    naip_chip = skimage.io.imread(naip_path)

                    # Load the Tx32x32 S2 file.
                    s2_images = skimage.io.imread(s2_path)
                    s2_chunks = np.reshape(s2_images, (-1, 32, 32, 3))

                    # SPECIAL CASE: when we are running a S2 upsampling experiment, sample 1 more 
                    # S2 image than specified. We'll use that as our "high res" image and the rest 
                    # as conditioning. Because the min number of S2 images is 18, have to use 17 for time series.
                    if self.datatype == 'just-s2':
                        rand_indices = random.sample(range(0, len(s2_chunks)), self.n_s2_images)
                        s2_chunks = [s2_chunks[i] for i in rand_indices[1:]]
                        s2_chunks = np.array(s2_chunks)
                        naip_chip = s2_chunks[0]  # this is a fake naip chip
                    else:
                        # Pick 18 random indices of s2 images to use.
                        rand_indices = random.sample(range(0, len(s2_chunks)), self.n_s2_images)
                        s2_chunks = [s2_chunks[i] for i in rand_indices]
                        s2_chunks = np.array(s2_chunks)

                        # Upsample to 512x512 (or whatever size your desired output is going to be.
                        up_s2_chunk = torch.permute(torch.from_numpy(s2_chunks), (0, 3, 1, 2))
                        up_s2_chunk = trans_fn.resize(up_s2_chunk, self.output_size, Image.BICUBIC, antialias=True)
                        s2_chunks = torch.permute(up_s2_chunk, (0, 2, 3, 1)).numpy()
                    break
                except:
                    print(s2_path)
                    hack += 1
                    index += hack

            # If conditioning on downsampled naip (along with S2), need to downsample original NAIP datapoint and upsample
            # it to get it to the size of the other inputs.
            if self.datatype == 's2_and_downsampled_naip':
                downsampled_naip = cv2.resize(naip_chip, dsize=(self.downsample_res,self.downsample_res), interpolation=cv2.INTER_CUBIC)
                downsampled_naip = cv2.resize(downsampled_naip, dsize=(self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

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

            elif self.datatype == 's2' or self.datatype == 'just-s2':

                if len(s2_chunks) == 1:
                    s2_chunk = s2_chunks[0]

                    [img_SR, img_HR] = Util.transform_augment(
				    [s2_chunk, naip_chip], split=self.split, min_max=(-1, 1))
                else:
                    [s2_chunks, img_HR] = Util.transform_augment(
                                    [s2_chunks, naip_chip], split=self.split, min_max=(-1, 1), multi_s2=True)

                    use_3d = False
                    if use_3d:
                        img_SR = torch.stack(s2_chunks)
                    else:
                        img_SR = torch.cat(s2_chunks)

            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

        elif self.datatype == 'naip':
            naip_path = self.datapoints[index]

            # Load the 512x512 NAIP chip.
            naip_chip = skimage.io.imread(naip_path)

            # Create the downsampled version on-the-fly.
            downsampled_naip = cv2.resize(naip_chip, dsize=(self.downsample_res,self.downsample_res), interpolation=cv2.INTER_CUBIC)
            downsampled_naip = cv2.resize(downsampled_naip, dsize=(self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

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
