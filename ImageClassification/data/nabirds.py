import os
import pandas as pd
import warnings
import numpy as np
import torch
from PIL import Image



from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, extract_archive


from torch.utils.data import DataLoader, Dataset



class NABirds(Dataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'nabirds/images'

    def __init__(self, root, train=True, transform=None):
        dataset_path = os.path.join(root, 'nabirds')
        self.root = root
        self.loader = default_loader
        self.train = train
        self.transform = transform

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}

def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names

def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents
