'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase, sampler=None):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            sampler=sampler,
            pin_memory=True)
    elif phase == 'val':
        print("val batch size:", dataset_opt['batch_size'])
        return torch.utils.data.DataLoader(
            dataset, batch_size=dataset_opt['batch_size'], shuffle=False, num_workers=dataset_opt['num_workers'], pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase, output_size=512, use_3d=False):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D

    l_res = None if not 'l_resolution' in dataset_opt else dataset_opt['l_resolution']
    r_res = None if not 'r_resolution' in dataset_opt else dataset_opt['r_resolution']
    n_s2_images = -1 if not 'n_s2_images' in dataset_opt else dataset_opt['n_s2_images']
    downsample_res = -1 if not 'downsample_res' in dataset_opt else dataset_opt['downsample_res']
    max_tiles = -1 if not 'max_tiles' in dataset_opt else dataset_opt['max_tiles']

    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=l_res,
                r_resolution=r_res,
                split=phase,
                need_LR=(mode == 'LRHR'),
                n_s2_images=n_s2_images,
                downsample_res=downsample_res,
                output_size=output_size,
                max_tiles=max_tiles,
                use_3d=use_3d
                )

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
