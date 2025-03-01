import torch
from privacy_account.comupute_dp_sgd import find_best_order
from torch.utils.data import DataLoader
from utils.Sampling_procedure import get_data_loaders_possion
from privacy_account.comupute_dp_sgd import apply_dp_sgd_analysis, apply_dp_sgd_analysis_epsilon_accumulation
from train_and_validation_privacy_account.train import train
from train_and_validation_privacy_account.train_with_dp import train_with_dp
from train_and_validation_privacy_account.validation import validation
from data.util.get_data import get_data
from model.get_model import get_model
from utils.dp_optimizer import get_dp_optimizer
from privacy_account import compute_rdp
import random
from sympy import symbols, exp, binomial, summation, log, Eq, solve
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import root
import time


def US_sampling(train_data, device, sample_ratio):
    '''
    q_1: the sample ratio
    :alpha: the order of the rdp parameter
    :param norms: the list of the norm of the gradient
    :order: the order of the renyi differential privacy, we calculate the optimal probability
    :param sample_set: the sample_set is probability of different sample probability of different clip threshold
    :sample_ratio: the ratio of being sampled
    :return: Rejection sample and the gradient cut
    '''

    fair_sample = []
    dataset = train_data.dataset
    tr_data = DataLoader(dataset, batch_size=1)
    # Get the size of the dataset from the DataLoader
    dataset_size = len(tr_data.dataset)

    # Generate iid_samples with size equal to the dataset size
    iid_samples = np.random.binomial(n=1, p=sample_ratio, size=dataset_size)
    successful_indices = np.where(iid_samples == 1)[0]
    dataset_accept = torch.utils.data.Subset(dataset, successful_indices)
    dataset_accept_1 = DataLoader(dataset_accept, batch_size=max(len(successful_indices), 1))
    return dataset_accept_1


def Fairness_US_DPSGD(estimated_number, start_time, train_data, test_data, model, optimizer, batch_size,
                      sample_ratio_ar, epsilon_budget, clip,
                      delta, sigma, device, fairness_parameter, training_order, rdp_order, file_location):
    # This step got the indices then utilize the indice.

    # The minibatch size is to got the sample ratio
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    orders = rdp_order
    iter = 1
    epsilon = 0.
    best_test_acc = 0.
    epsilon_list = []
    test_loss_list = []
    train_size = len(train_loader.dataset)
    test_size = len(test_dl.dataset)
    train_privacy = [0] * train_size
    test_privacy = [0] * test_size
    rdp_list = [0] * len(orders)
    file = open(file_location, 'a')
    sample_ratio_array = sample_ratio_ar
    count = 0
    while epsilon < epsilon_budget:
        program_start = time.time()
        if epsilon < epsilon_budget and epsilon < fairness_parameter and training_order == 0:  # unfairness training first, if the unfairness was not reachs to it's maxinmum then using the unfairness training process
            optimizer.fairness = 0
            optimizer.minibatch_size = batch_size[0]
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[0]
            else:
                sample_ratio = sample_ratio_array
        elif epsilon < epsilon_budget and epsilon <= epsilon_budget - fairness_parameter and training_order == 1:  # fairness training first, if the fairness not reaches to it's maximum, then keept fair training
            optimizer.fairness = 1
            optimizer.minibatch_size = batch_size[1]
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[1]
            else:
                sample_ratio = sample_ratio_array
        elif epsilon < epsilon_budget and epsilon >= fairness_parameter and training_order == 0:  # fair second,if the unfair reachs to it's minimum, then stop the unfair training
            optimizer.fairness = 1
            optimizer.minibatch_size = batch_size[1]
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[1]
            else:
                sample_ratio = sample_ratio_array
        elif epsilon < epsilon_budget and epsilon >= epsilon_budget - fairness_parameter and training_order == 1:  # fair first, then no need to be fair
            optimizer.fairness = 0
            optimizer.minibatch_size = batch_size[0]
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[0]
            else:
                sample_ratio = sample_ratio_array
        epsilon, order, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio, sigma, rdp_list, orders,
                                                                              delta)
        train_dl = US_sampling(train_loader, device, sample_ratio)  # possion sampling
        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)
        test_loss, test_accuracy = validation(model, test_dl, device)
        program_end = time.time()
        count = count + 1
        output_string_time = (f'one step training cost {(program_end - program_start) / 3600} hours, '
                              f'so far the program has complete {(count / estimated_number) * 100}%,'
                              f'so far the program has run {(program_end - start_time) / 3600} hours, '
                              f' the program will completed in {(program_end - program_start) / 3600 * (estimated_number - count)} hours.\n')
        print(output_string_time)
        file.write(output_string_time)
        output_string = (
            f'iters: {iter}, '
            f'epsilon: {epsilon:.4f} | '
            f'Test set: Average loss: {test_loss:.4f}, '
            f'Accuracy: ({test_accuracy:.2f}%)\n'
        )
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter
        epsilon_list.append(torch.tensor(epsilon))
        test_loss_list.append(test_loss)
        output_string = (
            f'iters: {iter}, '
            f'epsilon: {epsilon:.4f} | '
            f'Test set: Average loss: {test_loss:.4f}, '
            f'Accuracy: ({test_accuracy:.2f}%)\n'
        )
        file.write(output_string)
        print(
            f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter += 1

    print("------finished ------")
    file.close()
    return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]


if __name__ == '__main__':
    orders = [1 + x / 10.0 for x in range(1, 100)] +list(range(2, 64)) + [128, 256, 512]
    # datasets  has four options 'CIFAR-10', 'MNIST', 'IMDB', 'FMNIST'
    training_order = 0  # 0 represent unfairness training first, 1 represent fairness training first
    #this is the example of CIFAR-10
    dataset = 'CIFAR-10'
    sampling_ratio_unfair = 0
    sampling_ratio_fair = 0
    if dataset == 'IMDB' or dataset == 'CIFAR-10':
        sampling_ratio_unfair = 8192 / 50000
        sampling_ratio_fair = 19200 / 50000
        order = find_best_order(0, orders, 0.5, 3, 0.00001, 5.67, sampling_ratio_fair, sampling_ratio_unfair)
    else:
        sampling_ratio_unfair = 8192 / 60000
        sampling_ratio_fair = 19200 / 60000
    order = find_best_order(0, orders, 0.5, 3, 0.00001, 5.67, sampling_ratio_fair, sampling_ratio_unfair)
    momentum = 0.9
    input_norm = "BN"
    device = 'cpu'
    batch_size_fairness = 19200
    batch_size_unfairness = 8192
    delta = 0.00001
    train_data, test_data, dataset = get_data(dataset, augment=False)
    model = get_model('DPSGD', 'CIFAR-10', 'cpu')
    optimizer = get_dp_optimizer(dataset, 'DPSGD', 0.5, momentum, 0.1, 5.67, 8192, model, 0, 1)
    optimizer.microbatch_size=1
    test_accuracy, iter_1, best_test_acc, best_iter, model, [epsilon_list, test_loss_list] = Fairness_US_DPSGD(133,
                                                                                                               time.time(),
                                                                                                               train_data,
                                                                                                               test_data,
                                                                                                               model,
                                                                                                               optimizer,
                                                                                                               [batch_size_unfairness,batch_size_fairness],
                                                                                                               [sampling_ratio_unfair,sampling_ratio_fair],
                                                                                                               3, 0.1,
                                                                                                               delta,
                                                                                                               5.67,
                                                                                                               device,
                                                                                                               0, 1,
                                                                                                               order,
                                                                                                               'data/experiment_DP-SGD_CIFAR.txt')
