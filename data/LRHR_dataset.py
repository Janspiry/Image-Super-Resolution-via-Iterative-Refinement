from io import BytesIO
import torchvision
from osgeo import gdal
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import skimage.io
import os
import csv
import random
import cv2
import json
import glob
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import functional as trans_fn
import glob

totensor = torchvision.transforms.ToTensor()

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled.
    Source code: https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', need_LR=False,
                    n_s2_images=-1, downsample_res=-1, output_size=512, max_tiles=-1, use_3d=False, specify_val=True):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.need_LR = need_LR
        self.split = split
        self.n_s2_images = n_s2_images
        self.downsample_res = downsample_res
        self.output_size = output_size
        self.max_tiles = max_tiles
        self.use_3d = use_3d

        print("OUTPUT_SIZE:", self.output_size)
        print("DATAROOT:", dataroot)

        # WorldStrat case
        if datatype == 'worldstrat':
            self.all_bands = False
            self.use_3d = False

            # Hardcoded paths to data and splits
            self.splits_csv = '/data/piperw/worldstrat/dataset/stratified_train_val_test_split.csv'
            self.lr_path = '/data/piperw/worldstrat/dataset/dataset_download/zenodo-version/lr_dataset/'
            self.hr_path = '/data/piperw/worldstrat/dataset/dataset_download/zenodo-version/hr_dataset/'

            # Read in the csv file containing splits and filter out non-relevant images for this split.
            # Build a list of [hr_path, [lr_paths]] lists. 
            self.datapoints = []
            with open(self.splits_csv, newline='') as csvfile:
                read = csv.reader(csvfile, delimiter=' ')
                for i,row in enumerate(read):
                    # Skip the row with columns.
                    if i == 0:
                        continue

                    row = row[0].split(',')
                    tile = row[1]
                    split = row[-1]
                    if split != self.split:
                        continue

                    # A few paths are missing even though specified in the split csv, so skip them.
                    if not os.path.exists((os.path.join(self.lr_path, tile, 'L2A', tile+'-'+str(1)+'-L2A_data.tiff'))):
                        continue

                    # HR image for the current datapoint. Still using rgb as ground truth (instead of pansharpened).
                    hr_img_path = os.path.join(self.hr_path, tile, tile+'_rgb.png')

                    # Each HR image has 16 corresponding LR images.
                    lrs = []
                    for img in range(1, int(self.n_s2_images)+1):
                        lr_img_path = os.path.join(self.lr_path, tile, 'L2A', tile+'-'+str(img)+'-L2A_data.tiff')
                        lrs.append(lr_img_path)

                    self.datapoints.append([hr_img_path, lrs])
                print("Loaded ", len(self.datapoints), " WorldStrat datapoints.")
                self.data_len = len(self.datapoints)
            return

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
            self.naip_path = os.path.join(dataroot, 'naip_32')
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

        # NOTE: temporary code to train on just 1/100th of the available data
        #if self.split == 'train':
        #    self.naip_chips = random.sample(self.naip_chips, 11000)

        print("self.naip chips:", len(self.naip_chips), " self.naip_path:", self.naip_path)

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

    def get_tile_weight_sampler(self, tile_weights):
        weights = []
        for dp in self.datapoints:
            # Extract the NAIP chip from this datapoint's NAIP path.
            # With the chip, we can index into the tile_weights dict (naip_chip : weight)
            # and then weight this datapoint pair in self.datapoints based on that value.
            naip_path = dp[0]
            split = naip_path.split('/')[-1]
            chip = split[:-4]

            # If the chip isn't in the tile weights dict, then there weren't any OSM features
            # in that chip, so we can set the weight to be relatively low (ex. 1).
            if not chip in tile_weights:
                weights.append(1)
            else:
                weights.append(tile_weights[chip])

        print('using tile_weight_sampler, min={} max={} mean={}'.format(min(weights), max(weights), np.mean(weights)))
        #return torch.utils.data.WeightedRandomSampler(weights, len(self.datapoints))
        return CustomWeightedRandomSampler(weights, len(self.datapoints))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        # Classifier-free guidance, X% of the time we want to replace S2 images with black images 
        # for "unconditional" generation during training. 
        cfg = random.randint(0, 19)
        uncond = True if self.split == 'train' and cfg in [0,1,2,3] else False

        # Conditioning on S2, or S2 and downsampled NAIP.
        if self.datatype == 's2' or self.datatype == 's2_and_downsampled_naip' or self.datatype == 'just-s2':

            # A while loop and try/excepts to catch a few potential errors and continue if caught.
            counter = 0
            while True:
                index += counter  # increment the index based on what errors have been caught
                if index >= self.data_len:
                    index = 0

                datapoint = self.datapoints[index]
                naip_path, s2_path = datapoint[0], datapoint[1]

                # Load the 512x512 NAIP chip.
                naip_chip = skimage.io.imread(naip_path)

                # Check for black pixels (almost certainly invalid) and skip if found.
                if [0, 0, 0] in naip_chip:
                    counter += 1
                    #print(naip_path, " contains invalid pixels.")
                    continue

                # Load the T*32x32 S2 file.
                # There are a few bad S2 paths, if caught then skip to the next one.
                try:
                    s2_images = skimage.io.imread(s2_path)
                except:
                    print(s2_path, " failed to load correctly.")
                    counter += 1
                    continue

                # Reshape to be Tx32x32.
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
                    # Iterate through the 32x32 chunks at each timestep, separating them into "good" (valid)
                    # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
                    goods, bads = [], []
                    for i,ts in enumerate(s2_chunks):
                        if [0, 0, 0] in ts:
                            bads.append(i)
                        else:
                            goods.append(i)

                    # Pick 18 random indices of s2 images to use. Skip ones that are partially black.
                    if len(goods) >= self.n_s2_images:
                        rand_indices = random.sample(goods, self.n_s2_images)
                    else:
                        need = self.n_s2_images - len(goods)
                        rand_indices = goods + random.sample(bads, need)

                    s2_chunks = [s2_chunks[i] for i in rand_indices]
                    s2_chunks = np.array(s2_chunks)

                    # Convert to torch so we can do some reupsampling.
                    up_s2_chunk = torch.permute(torch.from_numpy(s2_chunks), (0, 3, 1, 2))

                    # Upsampling Option 1: Bicubic interpolation for the upsampling of the S2 chunks.
                    up_s2_chunk = trans_fn.resize(up_s2_chunk, self.output_size, Image.BICUBIC, antialias=True)

                    # Upsampling Option 2: Nearest Neighbor upsampling of the S2 chunks.
                    #up_s2_chunk = torch.repeat_interleave(up_s2_chunk, repeats=2, dim=2)
                    #up_s2_chunk = torch.repeat_interleave(up_s2_chunk, repeats=2, dim=3)

                    s2_chunks = torch.permute(up_s2_chunk, (0, 2, 3, 1)).numpy()
                break

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

                    if self.use_3d:
                        img_SR = torch.stack(s2_chunks)
                    else:
                        img_SR = torch.cat(s2_chunks)

            # Classifier-free guidance step, replace S2 images with all black images.
            if uncond:
                img_SR = torch.zeros_like(img_SR)

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

        elif self.datatype == 'worldstrat':
            hr_path, lr_paths = self.datapoints[index]

            # High res
            hr_im = skimage.io.imread(hr_path)[:, :, 0:3]
            hr_im = cv2.resize(hr_im, (640, 640)) # NOTE: temporarily downsizing the HR image to match the SR image
            hr_im = totensor(hr_im)
            img_HR = hr_im

	    # Load each of the LR images with gdal, since they're tifs.
            lr_ims = []
            for lr_path in lr_paths:
                raster = gdal.Open(lr_path)
                array = raster.ReadAsArray()

                # If all_bands is specified, trying to replicate exact WorldStrat methodology,
                # otherwise have option to run on RGB.
                if self.all_bands:
                    lr_im = array.transpose(1, 2, 0)
                    lr_im = self.lr_transform(lr_im)
                else:
                    lr_im = array.transpose(1, 2, 0)[:, :, 1:4]

                lr_ims.append(lr_im)

            if not self.all_bands:
                # Resize each Sentinel-2 image to the same spatial dimension.
                lr_ims = [totensor(cv2.resize(im, (640,640))) for im in lr_ims]

            img_LR = torch.stack(lr_ims, dim=0)
            if not self.use_3d:
                img_LR = torch.reshape(img_LR, (-1, 640,640))

            # Classifier-free guidance step, replace S2 images with all black images.
            if uncond:
                img_LR = torch.zeros_like(img_LR)

            return {'HR': img_HR, 'SR': img_LR, 'Index': index}

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
