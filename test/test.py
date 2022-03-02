import copy
import os
from re import template
import time
import sys
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR

sys.path.append("")
from optimizer import lambda_calculator
from model import resnet50, bag_tricks, classifier, diff_attention
from metric import cmc_map, re_ranking
from loss import id_loss, triplet_loss, center_loss, circle_loss, reg_loss
from data import transform, dataset, sampler
from util import config_parser, logger, tool, averager

if __name__ == '__main__':
    # 0 introduction
    print('Person Re-Identification')
    print('test')

    # 1 config and tools
    # 1.1 Get config.
    config = config_parser.get_config(sys.argv)
    # config_parser.print_config(config)
    # 1.2 Get logger.
    logger = logger.get_logger()
    logger.info('Finishing program initialization.')
    # 1.3 Set device.
    if config['basic']['device'] == 'CUDA':
        os.environ['CUDA_VISIBLE_DEVICES'] = config['basic']['gpu_id']
    if config['basic']['device'] == 'CUDA' and torch.cuda.is_available():
        use_gpu, device = True, torch.device('cuda:0')
        logger.info('Set GPU: ' + config['basic']['gpu_id'])
    else:
        use_gpu, device = False, torch.device('cpu')
        logger.info('Set cpu as device.')
    # 1.4 Set random seed.
    seed = config['basic'].getint('seed')
    tool.setup_random_seed(seed)

    # 3 data
    # 3 data
    dataset_style = config['dataset']['style']
    dataset_path = config['dataset']['path']
    verbose = config['dataset'].getboolean('verbose')
    height = config['dataset'].getint('height')
    width = config['dataset'].getint('width')
    size = (height, width)
    random_erasing = config['dataset'].getboolean('random_erasing')
    batch_size = config['dataset'].getint('batch_size')
    p = config['dataset'].getint('p')
    k = config['dataset'].getint('k')
    num_workers = config['dataset'].getint('num_workers')
    pin_memory = config['dataset'].getboolean('pin_memory')
    dataset_norm = config['dataset'].getboolean('norm')
    # 3.1 Get train set.
    train_path = os.path.join(dataset_path, 'bounding_box_train')
    train_transform = transform.get_transform(
        size=size, is_train=True, random_erasing=random_erasing)
    train_dataset = dataset.ImageDataset(
        style=dataset_style, path=train_path, transform=train_transform, name='Image Train', verbose=verbose)
    if p is not None and k is not None and p * k == batch_size:
        # Use triplet sampler.
        sampler = sampler.TripletSampler(
            labels=train_dataset.labels, batch_size=batch_size, p=p, k=k)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    # 3.2 Get query set.
    query_path = os.path.join(dataset_path, 'query')
    query_transform = transform.get_transform(size=size, is_train=False)
    query_dataset = dataset.ImageDataset(
        style=dataset_style, path=query_path, transform=query_transform, name='Image Query', verbose=verbose)
    query_loader = DataLoader(dataset=query_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory)
    # 3.3 Get gallery set.
    gallery_path = os.path.join(dataset_path, 'bounding_box_test')
    gallery_transform = transform.get_transform(size=size, is_train=False)
    gallery_dataset = dataset.ImageDataset(
        style=dataset_style, path=gallery_path, transform=gallery_transform, name='Image Gallery', verbose=verbose)
    gallery_loader = DataLoader(dataset=gallery_dataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=pin_memory)