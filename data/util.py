import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1), multi_s2=False, consistent_sizing=True):    

    # Sorta hacky edge case for when we have >1 S2 chunk. Expects S2 list to be first in img_list.
    imgs = []
    s2_len, other_len = 0, 0
    if multi_s2:
        s2_imgs = [totensor(img) for img in img_list[0]]
        s2_len = len(s2_imgs)
        other = [totensor(img) for img in img_list[1:]]
        other_len = len(other)
        imgs = s2_imgs + other
    else:
        imgs = [totensor(img) for img in img_list]

    # Added code to crop to 256x256, since some of the bedroom and church images are non-standard.
    # Only need this for LSUN experiments. Or other non-standard-sized datasets.
    if not consistent_sizing:
        if multi_s2:
            print("WARNING: consistent_sizing not implemented for S2 time series yet...")
        new_imgs = []
        for im in imgs:
            if not (im.shape[1] == 256 and im.shape[2] == 256):
                im = im[:, :256, :256]
            new_imgs.append(im[:, :256, :256])
        imgs = new_imgs

    # Apply flip augmentation to train data.
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)

    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]

    if multi_s2:
        new_ret_img = []
        s2_chunks = [i for i in ret_img[:s2_len]]

        if other_len == 1:
            new_ret_img = [s2_chunks, ret_img[s2_len]]
        elif other_len == 2:
            new_ret_img = [s2_chunks, ret_img[s2_len], ret_img[s2_len+1]]
        ret_img = new_ret_img

    return ret_img
