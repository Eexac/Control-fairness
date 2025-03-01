import time

import torch
from torch.utils.data import DataLoader
from utils.Sampling_procedure import get_data_loaders_possion
from privacy_account.comupute_dp_sgd import apply_dp_sgd_analysis, find_best_order
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


def constraint(px, alpha, g_norm, sigma, C0, m):
    """Privacy constraint function"""
    return m - compute_rdp(px, C0 * sigma / g_norm, 1, alpha)


def objective(px, alpha, g_norm, sigma, C0, m):
    """Objective function: minimize absolute privacy deviation"""
    return abs(constraint(px, alpha, g_norm, sigma, C0, m))  # Minimize absolute deviation


def Search(norm, q, order, sigma, C0, threshold_rdp):
    """Search for px that minimizes absolute deviation from DP constraint"""
    cons = {'type': 'ineq', 'fun': lambda px: constraint(px, order, norm, sigma, C0, threshold_rdp)}
    bounds = [(max(q, 1e-6), min(1, 0.9999))]
    initial_guess = (q + 1) / 2

    # Minimize absolute privacy deviation
    solution = minimize(objective, initial_guess, bounds=bounds, constraints=cons,
                        args=(order, norm, sigma, C0, threshold_rdp))

    return solution.x  # Best px found


def Fake_Fairness_sampling(train_data, q_1, order, clip, noise_multiplier, device, optimizer, model,
                           train_privacy):
    '''
    q_1: the sample ratio
    :alpha: the order of the rdp parameter
    :param norms: the list of the norm of the gradient
    :order: the order of the renyi differential privacy, we calculate the optimal probability
    :return: Rejection sample and the gradient cut
    '''
    model.train()
    norms = []
    dataset = train_data.dataset
    tr_data = DataLoader(dataset, batch_size=1)
    count = 0
    for id_1, (data, target) in enumerate(tr_data):  # calculate the gradient of the data
        count += 1
        data, target = data.to(device), target.to(device)
        # Zero the gradients
        optimizer.minibatch_size = len(data)
        optimizer.zero_grad()
        # Forward pass: Compute the model output
        output = model(data)
        # Compute the loss
        if output.shape == torch.Size([2]):
            output = output.unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(output, target)
        # Backward pass: Compute the gradients

        # The following code indicate the
        loss.backward()
        # Calculate the norm of the gradients
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        train_privacy[count - 1] += total_norm
        norms.append(total_norm / max(total_norm / clip, 1))
    gradient_norms = np.array(norms)
    fair_sample = []
    threshold_rdp = compute_rdp(q_1, noise_multiplier, 1, order)
    alpha = order
    for norm in norms:
        if norm >= clip:
            fair_sample.append(q_1)
        elif norm == 0:
            fair_sample.append(1)
        else:
            solution = Search(norm, q_1, alpha, noise_multiplier, clip, threshold_rdp)
            fair_sample.append(solution.x[0])
    iid_samples = np.random.binomial(n=1, p=fair_sample)
    successful_indices = np.where(iid_samples == 1)[0]
    dataset_accept = torch.utils.data.Subset(dataset, successful_indices)
    dataset_accept_1 = DataLoader(dataset_accept, batch_size=len(successful_indices))
    return dataset_accept_1





def Fake_fairness_DPSGD(estimated_number, start_time, train_data, test_data, model, optimizer, batch_size, sample_ratio,
                        epsilon_budget, clip, delta, sigma, device, rdp_order, file_location):
    # This step got the indices then utilize the indice.
    accumulated_norm = []  # this is used to store the norm of the  gradient
    # The minibatch size is to got the sample ratio
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
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
    file = open(file_location, 'a')
    count = 0
    optimizer.microbatch_size = 1
    best_iter = 0
    while epsilon < epsilon_budget:
        program_start = time.time()
        epsilon, best_alpha, rdp = apply_dp_sgd_analysis(sample_ratio, sigma, iter, orders,
                                                    delta)  # comupte privacy cost

        train_dl = Fake_Fairness_sampling(train_loader, sample_ratio, orders, clip, sigma, device, optimizer, model,
                                          train_privacy)  # possion sampling
        optimizer.minibatch_size = batch_size

        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)

        test_loss, test_accuracy = validation(model, test_dl, device)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter
        epsilon_list.append(torch.tensor(epsilon))
        test_loss_list.append(test_loss)
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

        file.write(output_string)
        print(
            f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter += 1
    print("------finished ------")
    file.close()
    return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]


if __name__ == '__main__':
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(2, 64)) + [128, 256, 512]
    # datasets  has four options 'CIFAR-10', 'MNIST', 'IMDB', 'FMNIST'
    dataset = 'IMDB'
    sampling_ratio = 0
    if dataset == 'IMDB' or dataset == 'CIFAR-10':
        sampling_ratio = 8192 / 50000
    else:
        sampling_ratio = 8192 / 60000
    order = find_best_order(1, orders, 3, 3, 0.00001, 5.67, sampling_ratio, sampling_ratio)
    momentum = 0.9
    device = 'cpu'
    batch_size = 8192
    delta = 0.00001
    train_data, test_data, dataset = get_data(dataset, augment=False)
    model = get_model('DPSGD', 'IMDB', 'cpu')
    optimizer = get_dp_optimizer(dataset, 'DPSGD', 0.5, momentum, 1, 5.67, batch_size, model, 0, 1)
    test_accuracy, iter_1, best_test_acc, best_iter, model, [epsilon_list, test_loss_list] = Fake_fairness_DPSGD(750,
                                                                                                                 time.time(),
                                                                                                                 train_data,
                                                                                                                 test_data,
                                                                                                                 model,
                                                                                                                 optimizer,
                                                                                                                 batch_size,
                                                                                                                 sampling_ratio,
                                                                                                                 3, 0.1,
                                                                                                                 delta,
                                                                                                                 5.67,
                                                                                                                 device,
                                                                                                                 order,'./data/experiment_IMDB.txt')
