import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, Subset
import numpy as np


class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):

            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]
            # the indice will be stored in a array, for each indice the where function will select the indice via the justification in ().
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations


class EquallySizedAndIndependentBatchSamplerWithoutReplace:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size, replace=False)

    def __len__(self):
        return self.iterations


class Top_k_Sampler:
    def __init__(self, dataset, k, minibatch_size, Gradient):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.Gradient = Gradient
        self.k = k

    def __iter__(self):
        top_k_indices = np.argsort(self.Gradient)[-self.k:][::-1]
        yield top_k_indices

    def __len__(self):
        return 1


class Importance_Sampler:
    # The iteration is set to 1. The importance sampling has two stages: rejection sampling and true sampling.    # The iteration is set to 1. The importance sampling has two stages: rejection sampling and true sampling.
    def __init__(self, dataset, minibatch_size, Rejection_sampling):
        '''
        :param dataset:
        :param minibatch_size:
        :param Rejection_sampling:
        :param model: the deep model
        :param optimizer: the optimizer that will be used to update the model
        '''
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.Rejection_sampling = Rejection_sampling
        self.iterations = 1  # Fixed to 1 as per the requirement

    def __iter__(self):  # This function makes the object iterable
        for _ in range(self.iterations):
            # First stage: Rejection sampling using Bernoulli trials
            rejection_indices = np.where(np.random.rand(self.length) < self.Rejection_sampling)[0]
            if len(rejection_indices) == 0:
                yield np.array([])  # Return an empty array if no samples
            else:
                yield rejection_indices  # Yield the indices obtained from true sampling

    def __len__(self):
        return self.length  # This returns the number of samples in the dataset


class EquallySizedAndIndependentBatchSamplerWithReplace:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size, replace=True)

    def __len__(self):
        return self.iterations


def get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations, drop_last=True):
    # 这个函数才是主要的采样
    def minibatch_loader(dataset):  # 具体调用这个函数的时候会给对应入参
        return DataLoader(
            dataset,  # 给定原本数据
            batch_sampler=EquallySizedAndIndependentBatchSamplerWithoutReplace(dataset, minibatch_size, iterations)
            # DataLoader中自定义从数据集中取样本的策略
        )

    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )

    return minibatch_loader, microbatch_loader


def get_data_loaders_uniform_with_replace(minibatch_size, microbatch_size, iterations, drop_last=True):
    def minibatch_loader(dataset):
        return DataLoader(
            dataset,
            batch_sampler=EquallySizedAndIndependentBatchSamplerWithReplace(dataset, minibatch_size, iterations)
        )

    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )

    return minibatch_loader, microbatch_loader


def get_data_loaders_possion(minibatch_size, microbatch_size, iterations, drop_last=True):
    def minibatch_loader(dataset):
        return DataLoader(
            dataset,
            batch_sampler=IIDBatchSampler(dataset, minibatch_size, iterations),
        )

    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )

    return minibatch_loader, microbatch_loader


# get data with top_k gradient norms
def get_data_loaders_top_k(minibatch_size, microbatch_size, Gradient, drop_last=True):
    def minibatch_loader(dataset):  # dataset, k, minibatch_size, Gradient
        return DataLoader(
            dataset,
            batch_sampler=Top_k_Sampler(dataset, minibatch_size, Gradient)
        )

    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )

    return minibatch_loader, microbatch_loader


def get_data_loaders_importance_sampling(minibatch_size, microbatch_size, rejection_sampling, drop_last=True):
    def minibatch_loader(dataset):
        # Use the Importance_Sampler to get the indices of the selected data points
        sampler = Importance_Sampler(dataset, minibatch_size, rejection_sampling)
        # Return a DataLoader for the selected data points
        return DataLoader(dataset, sampler=sampler, batch_size=1), sampler

    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )

    return minibatch_loader, microbatch_loader
