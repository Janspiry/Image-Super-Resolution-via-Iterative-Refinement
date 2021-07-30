from io import BytesIO

import lmdb
from PIL import Image
from numpy import not_equal
from torch.utils.data import Dataset
from torchvision import transforms


class LRHRDataset(Dataset):
    def __init__(self, dataroot, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_HR=True, need_LR=True):
        self.env = lmdb.open(dataroot, readonly=True, lock=False,
                             readahead=False, meminit=False)

        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_HR = need_HR
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
        return int(BytesIO(length))

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        with self.env.begin(write=False) as txn:
            if self.need_LR:
                lr_img_bytes = txn.get(
                    'lr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                img_LR = Image.open(BytesIO(lr_img_bytes))
                img_LR = self.transform(img_LR)
            if self.need_HR:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                img_HR = Image.open(BytesIO(hr_img_bytes))
                img_HR = self.transform(img_HR)
        return {'LR': img_LR, 'HR': img_HR, 'Index': index}
