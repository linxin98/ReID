import copy
import os
from re import template
import time
import sys
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR

sys.path.append("")
from optimizer import lambda_calculator
from model import resnet50, bag_tricks, classifier, diff_attention, agw
from metric import cmc_map, re_ranking
from loss import id_loss, triplet_loss, center_loss, circle_loss, reg_loss
from data import transform, dataset, sampler
from util import config_parser, logger, tool, averager, gaussian

if __name__ == '__main__':
    # 0 introduction
    print('Person Re-Identification')
    print('test')

    # 1 config and tools
    # 1.1 Get config.
    config = config_parser.get_config(sys.argv)
    config_parser.print_config(config)
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
    model_path = config['model']['path']
    num_class = config['model'].getint('num_class')
    num_feature = config['model'].getint('num_feature')
    bias = config['model'].getboolean('bias')
    in_transform = config['da']['in_transform']
    diff_ratio = config['da'].getint('diff_ratio')
    out_transform = config['da']['out_transform']
    aggregate = config['da'].getboolean('aggregate')
    diff_model_path = config['da']['diff_model_path']
    # 2.1 Get feature model.
    base_model = agw.Baseline()
    if use_gpu:
        base_model = base_model.to(device)
    base_model.load_state_dict(torch.load(model_path))
    logger.info('Base Model: ' + str(tool.get_parameter_number(base_model)))

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
    # if p is not None and k is not None and p * k == batch_size:
    #     # Use triplet sampler.
    #     sampler = sampler.TripletSampler(
    #         labels=train_dataset.labels, batch_size=batch_size, p=p, k=k)
    sampler = None
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

    # 4 get feature
    base_model.eval()
    val_start = time.time()
    with torch.no_grad():
        # Get train feature.
        logger.info('Load train data.')
        train_features = []
        train_pids = []
        train_camids = []
        for train_batch, (train_image, _, pids, camids) in enumerate(train_loader):
            if use_gpu:
                train_image = train_image.to(device)
            train_feature = base_model(train_image)
            train_feature = train_feature.detach().cpu().numpy()
            train_features.append(train_feature)
            train_pids.extend(pids)
            train_camids.extend(camids)
        train_features = np.concatenate(train_features, axis=0)
        print(train_features.shape, len(train_pids))
        # pca = PCA(n_components=0.9, random_state=seed)
        # pca.fit(train_features)
        # train_features = pca.transform(train_features)
        # print(train_features.shape, len(train_pids))
        # train_features = torch.cat(train_features, dim=0)
        # print(train_features.size(), train_features.dtype, train_features.device, len(train_pids))
        # # Get query feature.
        # logger.info('Load query data.')
        # query_features = []
        # query_pids = []
        # query_camids = []
        # for query_batch, (query_image, _, pids, camids) in enumerate(query_loader):
        #     if use_gpu:
        #         query_image = query_image.to(device)
        #     query_feature = base_model(query_image)
        #     query_feature = query_feature.detach().cpu().numpy()
        #     query_features.append(query_feature)
        #     query_pids.extend(pids)
        #     query_camids.extend(camids)
        # query_features = np.concatenate(query_features, axis=0)
        # print(query_features.shape, len(query_pids))
        # # Get gallery feature.
        # logger.info('Load gallery data.')
        # gallery_features = []
        # gallery_pids = []
        # gallery_camids = []
        # for gallery_batch, (gallery_image, _, pids, camids) in enumerate(gallery_loader):
        #     if use_gpu:
        #         gallery_image = gallery_image.to(device)
        #     gallery_feature = base_model(gallery_image)
        #     gallery_feature = gallery_feature.detach().cpu().numpy()
        #     gallery_features.append(gallery_feature)
        #     gallery_pids.extend(pids)
        #     gallery_camids.extend(camids)
        # gallery_features = np.concatenate(gallery_features, axis=0)
        # print(gallery_features.shape, len(gallery_pids))

    # 5 gmm
    time_start = time.time()
    # 5.1 Train cluster.
    logger.info('Start clustering.')
    # kmeans = MiniBatchKMeans(
    #     n_clusters=num_class, random_state=seed, batch_size=batch_size, verbose=1).fit(train_features)
    # kmeans = KMeans(
    #     n_clusters=num_class, random_state=seed, verbose=1).fit(train_features)
    dbscan = DBSCAN(eps=16, min_samples=4).fit(train_features)
    # gm = GaussianMixture(n_components=num_class, init_params='kmeans', n_init=1, covariance_type='full',
    #                      random_state=seed, verbose=2, verbose_interval=1).fit(train_features)
    # gm = gaussian.GaussianMixture(
    #     n_components=num_class, n_features=num_feature, init_params='random', covariance_type='full').to(device)
    # gm.fit(train_features)
    # 5.2 Remark feature.
    logger.info('Remarked.')
    # train_pred = kmeans.predict(train_features)
    train_pred = dbscan.labels_
    # train_pred = gm.predict(train_features)
    # train_pred = train_pred.detach().cpu().numpy()
    print(train_pred, train_pred.shape)
    # 5.3 Score cluster.
    score = metrics.adjusted_rand_score(train_pids, train_pred)
    logger.info("Adjusted Rand Score: {:.5}".format(score))
    time_end = time.time()
    logger.info('Time taken: ' + time.strftime("%H:%M:%S",
                                               time.gmtime(time_end - time_start)))
