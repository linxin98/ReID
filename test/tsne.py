import copy
import os
from re import template
import time
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    print('t-SNE for UDA')

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

    # 2 model
    market_model_path = '../result/20211108/[supervised bag]220725[base]120.pth'
    duke_model_path = '../result/20211129/[supervised bag]234045[base]120.pth'
    # 2.1 Get market model.
    market_model = bag_tricks.Baseline()
    if use_gpu:
        market_model = market_model.to(device)
    market_model.load_state_dict(torch.load(market_model_path))
    logger.info('Market Model: ' +
                str(tool.get_parameter_number(market_model)))
    # 2.2 Get duke model.
    duke_model = bag_tricks.Baseline()
    if use_gpu:
        duke_model = duke_model.to(device)
    duke_model.load_state_dict(torch.load(duke_model_path))
    logger.info('Duke Model: ' + str(tool.get_parameter_number(duke_model)))
    # 2.3 Get initial model.
    initial_model = bag_tricks.Baseline()
    if use_gpu:
        initial_model = initial_model.to(device)
    logger.info('Initial Model: ' +
                str(tool.get_parameter_number(initial_model)))

    # 3 data
    dataset_style = config['dataset']['style']
    market_path = '../dataset/Market-1501'
    duke_path = '../dataset/DukeMTMC-reID'
    verbose = config['dataset'].getboolean('verbose')
    height = config['dataset'].getint('height')
    width = config['dataset'].getint('width')
    size = (height, width)
    random_erasing = config['dataset'].getboolean('random_erasing')
    batch_size = 128
    p = 8
    k = 16
    num_workers = config['dataset'].getint('num_workers')
    pin_memory = config['dataset'].getboolean('pin_memory')
    dataset_norm = config['dataset'].getboolean('norm')
    # 3.1 Get market set.
    market_path = os.path.join(market_path, 'bounding_box_train')
    market_transform = transform.get_transform(
        size=size, is_train=True, random_erasing=random_erasing)
    market_dataset = dataset.ImageDataset(
        style=dataset_style, path=market_path, transform=market_transform, name='Image Train', verbose=verbose)
    if p is not None and k is not None and p * k == batch_size:
        # Use triplet sampler.
        market_sampler = sampler.TripletSampler(
            labels=market_dataset.labels, batch_size=batch_size, p=p, k=k)
    market_loader = DataLoader(dataset=market_dataset, batch_size=batch_size,
                               sampler=market_sampler, num_workers=num_workers, pin_memory=pin_memory)
    # 3.2 Get duke set.
    duke_path = os.path.join(duke_path, 'bounding_box_train')
    duke_transform = transform.get_transform(
        size=size, is_train=True, random_erasing=random_erasing)
    duke_dataset = dataset.ImageDataset(
        style=dataset_style, path=duke_path, transform=duke_transform, name='Image Train', verbose=verbose)
    if p is not None and k is not None and p * k == batch_size:
        # Use triplet sampler.
        duke_sampler = sampler.TripletSampler(
            labels=duke_dataset.labels, batch_size=batch_size, p=p, k=k)
    duke_loader = DataLoader(dataset=duke_dataset, batch_size=batch_size,
                             sampler=duke_sampler, num_workers=num_workers, pin_memory=pin_memory)
    # 3.3 Get batchs.
    batchs = 1
    market_batchs = []
    batch = 0
    for images, labels, _, _ in market_loader:
        batch += 1
        if batch > batchs:
            break
        if use_gpu:
            images = images.to(device)
            labels = labels.to(device)
        market_batchs.append((images, labels))
    duke_batchs = []
    batch = 0
    for images, labels, _, _ in duke_loader:
        batch += 1
        if batch > batchs:
            break
        if use_gpu:
            images = images.to(device)
            labels = labels.to(device)
        duke_batchs.append((images, labels))

    # 4 t-SNE
    save = config['train'].getboolean('save')
    save_per_epochs = config['train'].getint('save_per_epochs')
    save_path = config['train']['save_path']
    save_path = os.path.join(
        save_path, time.strftime("%Y%m%d", time.localtime()))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    market_model.eval()
    duke_model.eval()
    initial_model.eval()
    # 4.1 Initialise.
    feature_list = []
    label_list = []
    mode = []
    mode.append('market-initial')
    mode.append('market-market')
    mode.append('market-duke')
    # mode.append('duke-initial')
    # mode.append('duke-market')
    # mode.append('duke-duke')
    # 4.2 Get features and labels from batchs.
    with torch.no_grad():
        # market + initial model
        if 'market-initial' in mode:
            for images, labels in market_batchs:
                features = initial_model(images)
                feature_list.append(features)
                label_list.append(labels)
        # market + market model
        if 'market-market' in mode:
            for images, labels in market_batchs:
                labels = labels + 2000
                features = market_model(images)
                feature_list.append(features)
                label_list.append(labels)
        # market + duke model
        if 'market-duke' in mode:
            for images, labels in market_batchs:
                labels = labels + 4000
                features = duke_model(images)
                feature_list.append(features)
                label_list.append(labels)
        # duke + initial model
        if 'duke-initial' in mode:
            for images, labels in duke_batchs:
                labels = labels + 10000
                features = initial_model(images)
                feature_list.append(features)
                label_list.append(labels)
        # duke + market model
        if 'duke-market' in mode:
            for images, labels in duke_batchs:
                labels = labels + 12000
                features = market_model(images)
                feature_list.append(features)
                label_list.append(labels)
        # duke + duke model
        if 'duke-duke' in mode:
            for images, labels in duke_batchs:
                labels = labels + 14000
                features = duke_model(images)
                feature_list.append(features)
                label_list.append(labels)
    # 4.3 t-SNE.
    features = torch.cat(feature_list, dim=0)
    features = features.detach().cpu().numpy()
    labels = torch.cat(label_list, dim=0)
    labels = labels.detach().cpu().numpy()
    print(len(labels))
    _, labels = np.unique(labels, return_inverse=True)
    labels += 1
    print(labels)
    logger.info('Start t-SNE.')
    tsne_start = time.time()
    embedding = TSNE(n_components=2, init='pca',
                     random_state=seed).fit_transform(features)
    tsne_end = time.time()
    tsne_time = abs(tsne_start - tsne_end)
    logger.info('t-SNE time taken: ' +
                time.strftime("%H:%M:%S", time.gmtime(tsne_time)))
    # 4.4 Plot figure.
    fig = plt.figure(figsize=(15, 8))
    plt.title(' '.join(mode))
    ax = plt.gca()
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20')
    legend = ax.legend(*scatter.legend_elements(num=None), title="ID")
    ax.add_artist(legend)
    for i in range(embedding.shape[0]):
        if i % (k // 4) == 0:
            ax.text(embedding[i, 0], embedding[i, 1], labels[i])
    fig_save_name = '[t-SNE]{}[{}].jpg'.format(
        time.strftime("%H%M%S", time.localtime()), ' '.join(mode))
    plt.savefig(os.path.join(save_path, fig_save_name))
    plt.show()
