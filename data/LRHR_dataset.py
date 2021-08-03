from io import BytesIO

import lmdb
from PIL import Image
from numpy import not_equal
from torch.utils.data import Dataset
from torchvision import transforms


class LRHRDataset(Dataset):
    def __init__(self, dataroot, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.env = lmdb.open(dataroot, readonly=True, lock=False,
                             readahead=False, meminit=False)

        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        elif split == 'val':
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
                img_LR = self.transform(img_LR)

            hr_img_bytes = txn.get(
                'hr_{}_{}'.format(
                    self.r_res, str(index).zfill(5)).encode('utf-8')
            )
            img_HR = Image.open(BytesIO(hr_img_bytes))
            img_HR = self.transform(img_HR)

            sr_img_bytes = txn.get(
                'sr_{}_{}_{}'.format(
                    self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
            )
            img_SR = Image.open(BytesIO(sr_img_bytes))
            img_SR = self.transform(img_SR)
        if self.need_LR:
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
