from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from scipy.signal import convolve2d
import random
from torch.utils.data import Subset


class MovingMNIST(data.Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(
            self,
            root,
            train=True,
            split=1000,
            transform=None,
            target_transform=None,
            download=False, # You can set download=True for the first running
            seq_len=10,
            horizon=10,
            crop_size=None,
            downsample_size=None
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set
        self.seq_len = seq_len
        self.horizon = horizon
        self.crop_size = crop_size
        self.downsample_size = downsample_size
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def crop_image(self, img):
        """ img: [T, C, H, W]"""

        if self.crop_size is not None:
            return img[..., :self.crop_size, :self.crop_size]
        else:
            return img

    def downsample_image(self, img):
        """ img: [T, C, H, W]"""

        if self.downsample_size is not None:
            T, C, H, W = img.shape

            assert (
                    H // self.downsample_size != 0 or W // self.downsample_size != 0
            ), "downsampling rate cannot be divided by image size"

            #             h = H // self.downsample_size
            #             out = img.reshape(T, C, -1, self.downsample_size, h,
            #                               self.downsample_size).sum((-1, -3)) / self.downsample_size ** 2

            #             return out
            return img[..., ::self.downsample_size, ::self.downsample_size]
            # kernel = np.ones((self.downsample_size, self.downsample_size))
            # out = convolve2d(img, kernel, mode='valid')
            # return out[::self.downsample_size, ::self.downsample_size] / self.downsample_size ** 2
        else:
            return img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([self.transform(img), new_data],
                                                                                  dim=0)
            return new_data

        if self.train:
            seq = self.train_data[index, :self.seq_len]
            target = self.train_data[index, self.seq_len: (self.seq_len + self.horizon)]
        else:
            seq = self.test_data[index, :self.seq_len]
            target = self.test_data[index, self.seq_len:(self.seq_len + self.horizon)]
        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)

        seq = seq.unsqueeze(1)  # adding channel dimension
        target = target.unsqueeze(1)

        return seq / 255.0, target / 255.0

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        if self._check_exists():
            return
        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
        # process and save as torch files
        print('Processing...')
        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str