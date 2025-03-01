import os
from privacy_account.comupute_dp_sgd import find_best_order
import torch
from torch.utils.data import DataLoader, Subset
from utils.Sampling_procedure import get_data_loaders_possion
from privacy_account.comupute_dp_sgd import apply_dp_sgd_analysis, apply_dp_sgd_analysis_epsilon_accumulation
from train_and_validation_privacy_account.train import train
from train_and_validation_privacy_account.train_with_dp import train_with_dp
from train_and_validation_privacy_account.validation import validation
from data.util.get_data import get_data
from model.get_model import get_model
from utils.dp_optimizer import get_dp_optimizer
from privacy_account import compute_rdp, compute_eps
import random
from sympy import symbols, exp, binomial, summation, log, Eq, solve
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import root
import time
import copy
import pandas as pd


def privacy_accumulation_step(device, accumulation_indexes, fairness_parameter, order, train_data, model, optimizer,
                              train_privacy, q, clip, sigma):
    dataset_tr = train_data.dataset  # this statement requires that test_data is a dataloader object
    subset_dataset = Subset(dataset_tr, accumulation_indexes)
    tr_data_loader = DataLoader(subset_dataset, batch_size=1)
    count = 0
    model.train()
    for id_1, (data, target) in enumerate(tr_data_loader):  # calculate the gradient of the data
        count += 1
        data, target = data.to(device), target.to(device)
        # Zero the gradients
        optimizer.minibatch_size = len(data)
        optimizer.zero_grad()
        # Forward pass: Compute the model output
        output = model(data)
        # Compute the loss
        if output.shape == torch.Size([2]):  # necessary for IMDB dataset
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
        # print(total_norm)
        if total_norm <= 0.00001:
            rdp = [0] * len(order)
        elif total_norm > clip:
            rdp = compute_rdp(q, sigma, 1, order)
        else:
            rdp = compute_rdp(q, sigma * clip / total_norm, 1, order)

        train_privacy[id_1] += rdp[0]
        # this is privacy accumulation
    model.eval()


def privacy_account_end(train_privacy, privacy_current, file_path, orders, delta):
    privacy_cal = []
    print(f'The number of the data taken into privacy account is {len(train_privacy)}')

    # Compute epsilon for each data point and store it in privacy_cal
    for j in range(len(train_privacy)):
        epsilon, order = compute_eps(orders, train_privacy[j] + privacy_current, delta)
        privacy_cal.append(min(3.00, epsilon))

    # Convert privacy_cal to numpy array for easier processing
    privacy_cal = np.array(privacy_cal)

    # Count occurrences of each unique epsilon value
    unique_values, counts = np.unique(privacy_cal, return_counts=True)

    # Calculate percentage for each unique value
    percentages = (counts / len(train_privacy)) * 100  # Convert to percentage

    # Create a DataFrame with unique values and their corresponding percentages
    data = pd.DataFrame({
        'Epsilon': unique_values,
        'Percentage': percentages
    })

    # Save to Excel
    data.to_excel(file_path, index=False)


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


def Find_privacy_stop_criterion(orders, training_order, sample_ratio_array, sigma, epsilon,
                                Phi, delta):  # this program is to find the round to stop  the program
    stop_Phi = 0
    stop_final = 0
    privacy = 0.0
    rdp_list = [0] * len(orders)
    if training_order == 0:  # the unfair training first, then the fair training
        while privacy <= epsilon:
            stop_final += 1
            if privacy < Phi:
                privacy, order, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_array[0], sigma,
                                                                                      rdp_list, orders,
                                                                                      delta)
                stop_Phi += 1
            else:
                privacy, order, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_array[1], sigma,
                                                                                      rdp_list, orders,
                                                                                      delta)

    else:
        while privacy <= epsilon:
            stop_final += 1
            if privacy < epsilon - Phi:
                privacy, order, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_array[1], sigma,
                                                                                      rdp_list, orders,
                                                                                      delta)

            else:
                privacy, order, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_array[0], sigma,
                                                                                      rdp_list, orders,
                                                                                      delta)
                stop_Phi += 1
    return stop_Phi, stop_final


def Initial_gradient_clipping(dataset, indexes, optimizer, model, device, magnitude):
    dataset_tr = dataset.dataset  # this statement requires that test_data is a dataloader object
    subset_dataset = Subset(dataset_tr, indexes)
    tr_data_loader = DataLoader(subset_dataset, batch_size=1)
    count = 0
    model.train()
    norm_array = []
    for id_1, (data, target) in enumerate(tr_data_loader):  # calculate the gradient of the data
        count += 1
        data, target = data.to(device), target.to(device)
        # Zero the gradients
        optimizer.minibatch_size = len(data)
        optimizer.zero_grad()
        # Forward pass: Compute the model output
        output = model(data)
        # Compute the loss
        if output.shape == torch.Size([2]):  # necessary for IMDB dataset
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
        norm_array.append(total_norm)
    norm_array_np = np.array(norm_array)
    model.eval()
    return np.median(norm_array_np) * magnitude


def Fairness_DPSGD_US_Accounting(start_time, train_data, test_data, model, optimizer,
                                 batch_size, sample_ratio_ar, epsilon_budget, magnitude_clip, subset_size,
                                 delta, sigma, device, fairness_parameter, training_order, rdp_order, file_location,
                                 Output_directory):
    # This step got the indices then utilize the indice.
    # Training order must be 0 unfair training first then fair training
    # The minibatch size is to got the sample ratio
    File_name = Output_directory + f'/PHI_{str(fairness_parameter).replace(".", "_")}_size_{str(subset_size).replace(".", "_")}_magnitude_{str(magnitude_clip).replace(".", "_")}_PRIVACY_account.xlsx'
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    orders = rdp_order
    stop_Phi = 0
    stop_final = 0
    privacy_before = [0] * len(orders)
    privacy_starter = []
    epsilon_temp = 0
    stop_Phi, stop_final = Find_privacy_stop_criterion(orders, training_order, sample_ratio_ar, sigma,
                                                       epsilon_budget,
                                                       fairness_parameter, delta)
    print(stop_Phi, stop_final)
    privacy_before = [0] * len(orders)
    while epsilon_temp <= fairness_parameter:  # unfairness training
        epsilon_temp, best_alpha, privacy_before = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_ar[0],
                                                                                              sigma,
                                                                                              privacy_before,
                                                                                              orders,
                                                                                              delta)  # comupte privacy cost
    privacy_current = copy.deepcopy(privacy_before)
    while epsilon_temp <= epsilon_budget:
        epsilon_temp, best_alpha, privacy_before = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_ar[1],
                                                                                              sigma,
                                                                                              privacy_before,
                                                                                              orders,
                                                                                              delta)  # comupte privacy cost
    privacy_Phi = privacy_before[0] - privacy_current[0]
    # this is the fairness training privacy loss, because the privacy loss in fairnees training is known to be uniform, we do not need to train the model in practice instead we only need to accumulate it virtually
    iter = 1
    epsilon = 0.
    best_test_acc = 0.
    epsilon_list = []
    test_loss_list = []
    train_size = len(train_loader.dataset)
    # this is the indices that we're tracking the privacy budget of each datapoint
    accumulation_indices = random.sample(range(train_size), subset_size)
    privacy_accumulation = np.zeros((subset_size))
    # this is privacy initialization
    file = open(file_location, 'a')
    test_accuracy = 0
    sample_ratio_array = sample_ratio_ar
    count = 0
    best_iter = 0
    clip = Initial_gradient_clipping(train_loader, accumulation_indices, optimizer, model, device, magnitude_clip)
    print(f'The clip bound is: {clip}')
    optimizer.clip = clip
    count = 0
    while count < stop_Phi:
        program_start = time.time()
        count = count + 1
        privacy_accumulation_step(device, accumulation_indices, fairness_parameter, orders, train_loader, model,
                                  optimizer,
                                  privacy_accumulation, sample_ratio_array[0], clip, sigma)
        train_dl = US_sampling(train_loader, device, sample_ratio_array[0])  # possion sampling
        for id, (data, target) in enumerate(train_dl):  # this step must before the training step
            optimizer.minibatch_size = batch_size
            optimizer.fainess = 0  # unfair training
        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)
        test_loss, test_accuracy = validation(model, test_dl, device)
        program_end = time.time()
        output_string_time = (f'one step training cost {(program_end - program_start) / 3600} hours, '
                              f'so far the program has complete {(count / stop_Phi) * 100}%,'
                              f'so far the program has run {(program_end - start_time) / 3600} hours, '
                              f' the program will completed in {(program_end - program_start) / 3600 * (stop_Phi - count)} hours.\n')
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
    privacy_account_end(privacy_accumulation, privacy_current, File_name, orders, delta)
    print("------finished ------")
    file.close()
    return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]


if __name__ == '__main__':
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    datasets = ['CIFAR-10', 'MNIST', 'IMDB', 'FMNIST']
    momentum = 0.9
    input_norm = "BN"
    # device_1 = torch.device('cuda:0')
    device_1 = 'cpu'
    batch_size = 19200
    delta = 0.00001
    for magnitude in [0.1, 1.5]:
        for data_iter in datasets:
            for Phi in [0.5, 1, 1.5, 2]:
                if data_iter == 'IMDB' or data_iter == 'CIFAR-10':
                    sample_rat = 19200 / 50000
                    order = find_best_order(0, orders, 0.5, 3, 0.00001, 5.67, sample_rat, sample_rat)
                else:
                    sample_rat = 19200 / 60000
                    order = find_best_order(0, orders, 0.5, 3, 0.00001, 5.67, sample_rat, sample_rat)
                train_data, test_data, dataset = get_data(data_iter, augment=False)
                model = get_model('DPSGD', data_iter, device_1)
                optimizer = get_dp_optimizer(data_iter, 'DPSGD', 0.5, momentum, 1, 5.67, batch_size, model, 0, 1)
                optimizer.microbatch_size = 1
                dir = f'./data/{dataset}/Privacy_account/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                test_accuracy, iter_1, best_test_acc, best_iter, model, [epsilon_list,
                                                                         test_loss_list] = Fairness_DPSGD_US_Accounting(
                    time.time(),
                    train_data,
                    test_data,
                    model,
                    optimizer,
                    batch_size,
                    [sample_rat,
                     sample_rat],
                    3.0, magnitude, 3000,
                    delta,
                    5.67,
                    device_1,
                    Phi, 1, order,
                    f'./data/{dataset}//Privacy_account/experiment_DP-SGD_Utility.txt', dir)
