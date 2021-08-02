import argparse
from io import BytesIO
import multiprocessing
from functools import partial
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb


def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    mini_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    lr_img = resize_and_convert(mini_img, sizes[1], resample)

    if lmdb_save:
        mini_img = image_convert_bytes(mini_img)
        hr_img = image_convert_bytes(hr_img)
        lr_img = image_convert_bytes(lr_img)

    return [mini_img, hr_img, lr_img]


def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    return img_file.name.split('.')[0], out


def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)

    files = [p for p in Path(f'{img_path}').glob(f'**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/mini_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/lr_{}_{}'.format(out_path,
                    sizes[0], sizes[1]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    total = 0
    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            mini_img, hr_img, lr_img = imgs
            if not lmdb_save:
                mini_img.save(
                    '{}/mini_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                hr_img.save(
                    '{}/hr_{}/{}.png'.format(out_path, sizes[1], i.zfill(5)))
                lr_img.save(
                    '{}/lr_{}_{}/{}.png'.format(out_path, sizes[0], sizes[1], i.zfill(5)))
            else:
                with env.begin(write=True) as txn:
                    txn.put('mini_{}_{}'.format(
                        sizes[0], i.zfill(5)).encode('utf-8'), mini_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('lr_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), lr_img)
            total += 1
        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='../dataset/ffhq_128')
    parser.add_argument('--out', '-o', type=str, default='../dataset/ffhq')

    parser.add_argument('--size', type=str, default='16,128')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--resample', type=str, default='bicubic')
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)
