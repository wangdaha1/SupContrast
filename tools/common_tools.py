import numpy as np
import os
import os.path
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import CIFAR10, CIFAR100

# we need to get the absolute paths of imgs in these standard datasets
class CIFAR10_withpath(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        self.absolute_paths = []  # add

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.absolute_paths.extend(entry['filenames'])  # [b'batch_label', b'labels', b'data', b'filenames']
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])


        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, absolute_path) where target is index of the target class.
        """
        img, target, path = self.data[index], self.targets[index], self.absolute_paths[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path


class CIFAR100_withpath(CIFAR10_withpath):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class Train_Dataset(Dataset):
    '''For creating a new dataset for training
    '''
    def __init__(self, data, noisy_labels, clean_labels, transform=None, target_transform=None):
        self.train_data = np.array(data)
        self.noisy_train_labels = np.array(noisy_labels)
        self.clean_train_labels = np.array(clean_labels)
        self.length = len(self.noisy_train_labels)
        self.target_transform = target_transform

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        img, target, clean_label = self.train_data[index], self.noisy_train_labels[index], self.clean_train_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            clean_label = self.target_transform(clean_label)

        return img, target, clean_label

    def __len__(self):
        return self.length

    def getData(self):
        return self.train_data, self.noisy_train_labels, self.clean_train_labels

