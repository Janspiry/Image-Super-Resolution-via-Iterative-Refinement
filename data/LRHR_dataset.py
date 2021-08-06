from io import BytesIO
import lmdb
from PIL import Image
from numpy import not_equal
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch


class LRHRDataset(Dataset):
    def __init__(self, dataroot, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.env = lmdb.open(dataroot, readonly=True, lock=False,
                             readahead=False, meminit=False)

        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        if self.data_len > 0:
            return self.data_len
        with self.env.begin(write=False) as txn:
            length = txn.get("length".encode("utf-8"))
        return int(length)

    def AugmentWithTransform(self, img_list, hflip=True, rot=False):
        # horizontal flip OR rotate
        hflip = hflip and (self.split == 'train' and random.random() < 1)
        vflip = rot and (self.split == 'train' and random.random() < 1)
        rot90 = rot and (self.split == 'train' and random.random() < 1)

        def _augment(img):
            if hflip:
                img = torch.flip(img, dims=[2])
            if vflip:
                img = torch.flip(img, dims=[1])
            if rot90:
                img = img.permute([0, 2, 1])
            return img

        return [_augment(self.transform(img)) for img in img_list]

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        with self.env.begin(write=False) as txn:
            if self.need_LR:
                lr_img_bytes = txn.get(
                    'lr_{}_{}'.format(
                        self.l_res, str(index).zfill(5)).encode('utf-8')
                )
                img_LR = Image.open(BytesIO(lr_img_bytes))
            hr_img_bytes = txn.get(
                'hr_{}_{}'.format(
                    self.r_res, str(index).zfill(5)).encode('utf-8')
            )
            img_HR = Image.open(BytesIO(hr_img_bytes))

            sr_img_bytes = txn.get(
                'sr_{}_{}_{}'.format(
                    self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
            )
            img_SR = Image.open(BytesIO(sr_img_bytes))
        if self.need_LR:
            [img_LR, img_SR, img_HR] = self.AugmentWithTransform(
                [img_LR, img_SR, img_HR])
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = self.AugmentWithTransform(
                [img_SR, img_HR])
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
