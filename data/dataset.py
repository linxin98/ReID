import copy
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from torch.functional import _return_inverse
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.dataset import T


class ImageDataset(Dataset):
    def __init__(self, style, path, transform, name):
        super(ImageDataset, self).__init__()
        # dataset parameters
        self.style = style
        self.path = path
        self.transform = transform
        self.name = name
        # dataset variables
        self.length = 0
        self.available_index = []
        self.images = []
        self.pids = []
        self.camids = []
        self.labels = []
        self.true_labels = []
        # dataset all variables
        self.all_length = 0
        self.all_images = []
        self.all_pids = []
        self.all_camids = []
        self.all_labels = []
        self.all_true_labels = []
        # Initialize dataset.
        self.initialize_dataset()

    def __len__(self):
        return self.length

    def load_item(self, item, style):
        if style == 'market':
            pattern = re.compile(r'([-\d]+)_c(\d)')
        # Check item is a file.
        allow_type = ['jpg', 'png']
        if not item[-3:] in allow_type:
            return None
        pid, camid = map(int, pattern.search(item).groups())
        if pid == -1 or pid == 0:
            return None
        return pid, camid

    def initialize_dataset(self):
        # Load folder.
        files = os.listdir(self.path)
        files.sort()
        for file in files:
            results = self.load_item(file, self.style)
            # Add item into dataset.
            if results is not None:
                self.all_images.append(file)
                self.all_pids.append(results[0])
                self.all_camids.append(results[1])
        self.all_length = len(self.all_images)
        _, labels = np.unique(self.all_pids, return_inverse=True)
        self.all_labels = copy.deepcopy(labels)
        self.all_true_labels = copy.deepcopy(labels)
        # Create variables for calling.
        self.length = self.all_length
        self.available_index = [i for i in range(self.all_length)]
        self.set_available()

    def summary_dataset(self):
        print('=' * 25)
        print("Dataset Summary:", self.name)
        print('      {:>9}/{:>7}'.format('available', 'all'))
        print('#image: {:7d}/{:7d}'.format(self.length, self.all_length))
        print('#pid:   {:7d}/{:7d}'.format(len(np.unique(self.pids)),
                                            len(np.unique(self.all_pids))))
        print('#camid: {:7d}/{:7d}'.format(len(np.unique(self.camids)),
                                            len(np.unique(self.all_camids))))
        print('=' * 25)

    def set_available(self, available_index=None):
        if available_index is not None:
            self.available_index = available_index
            self.length = len(available_index)
        self.images = [self.all_images[index]
                       for index in self.available_index]
        self.pids = [self.all_pids[index]
                     for index in self.available_index]
        self.camids = [self.all_camids[index]
                       for index in self.available_index]
        self.labels = [self.all_labels[index]
                       for index in self.available_index]
        self.true_labels = [self.all_true_labels[index]
                            for index in self.available_index]

    def set_labels(self, new_labels):
        self.all_labels = new_labels
        self.labels = [self.all_labels[index]
                       for index in self.available_index]
        
    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        file = self.images[index]
        pid = self.pids[index]
        camid = self.camids[index]
        label = self.labels[index]
        # Load image.
        file_path = os.path.join(self.path, file)
        image = Image.open(file_path)
        # Convert image to tenser.
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, pid, camid


class FeatureDataset(Dataset):
    def __init__(self, origin_dataset, model, device, batch_size, norm, num_workers, pin_memory):
        super(FeatureDataset, self).__init__()
        # dataset parameters
        self.origin_dataset = origin_dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.norm = norm
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # dataset variables
        self.features = []
        # dataset all variables
        self.all_features = []
        # Preprocess feature.
        self.detect_feature()

    def __len__(self):
        return self.origin_dataset.length

    def detect_feature(self):
        dataloader = DataLoader(self.origin_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=self.pin_memory)
        print('Detect feature from the origin dataset.')
        with torch.no_grad():
            self.model.eval()
            batch = 0
            for images, _, _, _ in dataloader:
                batch += 1
                if batch % 50 == 0:
                    print('Batch: {}'.format(batch))
                images = images.to(self.device)
                features = self.model(images)
                if self.norm:
                    features = torch.nn.functional.normalize(
                        features, p=2, dim=1)
                features = features.detach().cpu()
                for i in range(features.shape[0]):
                    self.all_features.append(features[i])
        print('Finish detecting feature.')
        self.features = [self.all_features[index]
                         for index in self.origin_dataset.available_index]

    def summary_dataset(self):
        self.origin_dataset.summary_dataset()

    def set_available(self, available_index=None):
        self.origin_dataset.set_available(available_index)
        self.features = [self.all_features[index]
                         for index in self.origin_dataset.available_index]

    def set_labels(self, new_labels):
        self.origin_dataset.set_labels(new_labels)
        
    def get_labels(self):
        return self.origin_dataset.labels

    def __getitem__(self, index):
        feature = self.features[index]
        pid = self.origin_dataset.pids[index]
        camid = self.origin_dataset.camids[index]
        label = self.origin_dataset.labels[index]
        return feature, label, pid, camid


if __name__ == '__main__':
    style = 'market'
    path = '../dataset/market/bounding_box_train'
    dateset = ImageDataset(style, path, None, 'Train')
    dateset.summary_dataset()
    # for i in range(100, 120):
    #     print(dateset[i])
