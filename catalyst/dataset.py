import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from functools import reduce
import operator

import numpy as np

import utils_ as utils

class RandomDataset(Dataset):
    def __init__(self, N, d, std, noise_type="gaussian", projection_dim=None, projection_more_noise_ratio=0.1):
        super(RandomDataset, self).__init__()
        self.d = d
        self.d_total = reduce(operator.mul, d, 1) 
        self.std = std
        self.N = N
        self.noise_type = noise_type
        self.projection_dim = projection_dim
        self.projection_more_noise_ratio = projection_more_noise_ratio
        self.regenerate()

    def regenerate(self):
        self.x = torch.FloatTensor(self.N, *self.d)
        if self.noise_type == "gaussian":
            self.x.normal_(0, std=self.std) 
        elif self.noise_type == "uniform":
            self.x.uniform_(-self.std / 2, self.std / 2)
        else:
            raise NotImplementedError(f"Unknown noise type: {self.noise_type}")

        if self.projection_dim is not None and self.projection_dim < self.d_total:
            # Project the dataset into a random low-dimensional space
            # create an orthonomial projection
            tmp = torch.FloatTensor(self.d_total, self.projection_dim).normal_(0, 1)
            q, r = tmp.qr(some=False)
            # Take first projection_dim subspace.
            self.q = q[:, :self.projection_dim]
            self.x = (self.x.view(self.N, -1) @ self.q) @ self.q.t()
            self.x = self.x.view(self.N, *self.d)
            # add noise as well. 
            self.x += torch.zeros_like(self.x).normal_(0, self.std * self.projection_more_noise_ratio)

    def __getitem__(self, idx):
        return self.x[idx], -1

    def __len__(self):
        return self.N

def init_dataset(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,), (0.5,))]) 

    if args.use_data_aug: 
        transform_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_cifar10_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_cifar10_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "gaussian" or args.dataset == "uniform":
        if args.use_cnn:
            d = (1, 16, 16)
        else:
            d = (args.data_d,)
        d_output = 100
        train_dataset = RandomDataset(
                args.random_dataset_size, d, args.data_std, 
                noise_type=args.dataset, 
                projection_dim=args.projection_dim, 
                projection_more_noise_ratio=args.projection_more_noise_ratio)

        # eval_dataset = RandomDataset(10240, d, args.data_std, noise_type=args.dataset, projection_dim=args.projection_dim)
        eval_dataset = RandomDataset(10240, d, args.data_std, noise_type=args.dataset)

    elif args.dataset == "mnist":
        train_dataset = datasets.MNIST(
                root=args.data_dir, train=True, download=True, 
                transform=transform)

        eval_dataset = datasets.MNIST(
                root=args.data_dir, train=False, download=True, 
                transform=transform)

        d = (1, 28, 28)
        d_output = 10

    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
                root=args.data_dir, train=True, download=True, 
                transform=transform_cifar10_train)

        eval_dataset = datasets.CIFAR10(
                root=args.data_dir, train=False, download=True, 
                transform=transform_cifar10_test)

        if not args.use_cnn:
            d = (3 * 32 * 32, )
        else: 
            d = (3, 32, 32)
        d_output = 10

    else:
        raise NotImplementedError(f"The dataset {args.dataset} is not implemented!")

    return d, d_output, train_dataset, eval_dataset
