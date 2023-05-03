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
                print("k,v:", k, v)
                naip_name, naip_chip = k[:-12], k[-11:]
                naip_path = os.path.join(self.naip_path, 'tci', naip_chip+'.png')

                s2_list = [os.path.join(self.s2_path, x[0], 'tci', x[1]+'_'+x[2]) for x in v]
                s2_paths = random.sample(s2_list, self.n_s2_images)
                print("naip path:", naip_path)
                print("s2 list:", s2_list)
                print("s2 chosen paths:", s2_paths)

                self.datapoints.append([naip_path, s2_paths])
                break

            self.data_len = len(self.naip_chips)
            print("number of naip chips:", self.data_len, " & len(meta):", len(self.meta))

        # Added code for S2/NAIP specific format.
        # Option to uncomment code below to also condition on downsampled NAIP.
        elif datatype == 'satellite_imagery':
            self.sr_path = '/data/s2_naip_pairs/s2/'
            self.hr_path = '/data/s2_naip_pairs/naip/'

            self.naip_chips = glob.glob(self.hr_path + "/**/*.png", recursive=True)

            meta_file = open('/data/s2_naip_pairs/metadata/naip_to_s2.json')
            self.meta = json.load(meta_file)

            # To be safe, get rid of any naip chip entries that do not exist in the meta file,
            # as well as naip chip entires with a 0-length list of corresponding S2 tiles.
            print("Length of meta before deleting empty lists...", len(self.meta))
            for k,v in list(self.meta.items()):
                if len(v) < 1:
                    del self.meta[k]
                    self.naip_chips.remove(k)

            self.data_len = len(self.naip_chips)
            print("number of naip chips:", self.data_len)
            print("number of meta entries:", len(self.meta))

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

        # Conditioning on S2.
        if self.datatype == 's2':
            datapoint = self.datapoints[index]
            naip_path, s2_paths = datapoint[0], datapoint[1]

            if not os.path.exists(naip_path):
                print("WARNING: ", naip_path, "does not exist...")

            # Load the 512x512 NAIP chip.
            naip_chip = skimage.io.imread(naip_path)

            

        # Added code to work with the custom S2/NAIP imagery, and having to extract and resize things.
        elif self.datatype == 'satellite_imagery':

            # Hacky hacky while loop to catch a couple edge cases during training.
            count = 0
            while True:
                index += count

                # Load in the 512x512 NAIP chip.
                # ex. /data/s2_naip_pairs/naip/m_3810131_sw_14_060_20190709/tci/28678_50314.png
                naip_chip = self.naip_chips[index]
                if not os.path.exists(naip_chip):
                    print("WARNING: ", naip_chip, " does not exist...")
                    count += 1
                    continue

                img_HR = skimage.io.imread(naip_chip)

                ### NOTE: comment this section out if you do NOT want to condition on downsampled NAIP
                # This downsamples the NAIP image by some amount and then upsamples it to get it back to size.
                #downsampled_naip = cv2.resize(img_HR, dsize=(4,4), interpolation=cv2.INTER_CUBIC)
                #downsampled_naip = cv2.resize(downsampled_naip, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

                # Extract components from the NAIP chip filepath.
                split = naip_chip.split('/')
                chip = split[-1][:-4]
                img_name = split[4]
                date = img_name[-8:-2]

                s2_list = self.meta[img_name + '_' + chip]

                # Find the S2 tile that contains the chip. 
                tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16  # s2 tile that contains the naip chip
                format_tile = str(tile[0]) + '_' + str(tile[1])

                # The meta naip-to-S2 dict should only contain entries with S2 lists of length > 0.
                # So iterate over the list of S2 tiles and find the one closest temporally to the NAIP chip.
                naip_y, naip_m, naip_d = date[:4], date[4:6], date[6:]
                best_option = s2_list[0]   # TODO: hmmm maybe the default should be None so we don't accidentally give it the wrong geospatial located tile
                best_diff = 1000
                for s2_opt in s2_list:
                    t,d = s2_opt
                    if t == format_tile:  # only want to consider S2 tiles that overlap with this NAIP chip
                        
                        if d[:4] == naip_y:
                            if d[4:6] == naip_m:  # if an image from the same month as naip exists, then just use it and break out of loop
                                best_option = s2_opt
                                break

                            diff = abs(int(d[4:6]) - int(naip_m))
                            if diff < best_diff:
                                best_diff = diff
                                best_option = s2_opt
                        else:
                            if best_diff == 1000:  # if we haven't yet found the correct S2 tile from any time period, at least save this one from a different year
                                best_option = s2_opt
                                best_diff = 999

                if best_diff == 1000:  # if we made it through the whole loop without finding the tile, then it doesn't exist. hopefully this doesn't happen
                    print("WARNING: did not find a matching tile....", format_tile)
                    count += 1
                    continue

                best_opt_d = best_option[1]
                format_date = best_opt_d[:4] + '-' + best_opt_d[4:6]

                s2_tile = os.path.join(self.sr_path, format_date, 'tci', format_tile + '.png')
                if not os.path.exists(s2_tile):
                    print("WARNING: ", s2_tile, " does not exist :(")
                    count += 1
                    continue
                else:
                    s2_img = skimage.io.imread(s2_tile)

                # Now that we have the S2 tile loaded, we need to extract the 32x32 chunk out of it, based on the NAIP chip.
                s2_left_corner = tile[0] * 16, tile[1] * 16
                diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]
                s2_chunk = s2_img[diffs[1]*32 : (diffs[1]+1)*32, diffs[0]*32 : (diffs[0]+1)*32, :]

                img_SR = torch.permute(torch.from_numpy(s2_chunk), (2, 0, 1))
                img_SR = trans_fn.resize(img_SR, 512, Image.BICUBIC)
                img_SR = trans_fn.center_crop(img_SR, 512)
                img_SR = torch.permute(img_SR, (1, 2, 0)).numpy()

                ### NOTE: comment this is you do NOT want to condition with both S2 and downsampled NAIP
                #[img_SR, downsampled_naip, img_HR] = Util.transform_augment(
                #                                [img_SR, downsampled_naip, img_HR], split=self.split, min_max=(-1, 1))
                #img_SR = torch.cat((img_SR, downsampled_naip))


                [img_SR, img_HR] = Util.transform_augment(
                                        [img_SR, img_HR], split=self.split, min_max=(-1, 1))

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
