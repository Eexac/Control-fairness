import time

import torch
from torch.utils.data import DataLoader

from privacy_account.comupute_dp_sgd import apply_dp_sgd_analysis_epsilon_accumulation
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
import pandas as pd
from scipy.optimize import root


def Possion_sampling(train_data, device, sample_ratio):
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
    print('the size of dataset is' + str(dataset_size))
    # Generate iid_samples with size equal to the dataset size
    iid_samples = np.random.binomial(n=1, p=sample_ratio, size=dataset_size)
    successful_indices = np.where(iid_samples == 1)[0]
    dataset_accept = torch.utils.data.Subset(dataset, successful_indices)
    dataset_accept_1 = DataLoader(dataset_accept, batch_size=max(len(successful_indices), 1))
    return dataset_accept_1


def find_closest_value_and_index(clip_set, actual_gradient_norm):
    # Convert clip_set to a numpy array (if it is not already)
    clip_set = np.array(clip_set)

    # Compute the absolute differences between each value in clip_set and actual_gradient_norm
    differences = np.abs(clip_set - actual_gradient_norm)

    # Find the index of the minimum difference
    closest_index = np.argmin(differences)

    # Return the closest value and its index
    return clip_set[closest_index], closest_index


def equation(px, alpha, g_norm, sigma, C0, m):
    # Return the equation that should equal zero
    return m - compute_rdp(px, C0 * sigma / g_norm, 1, alpha)


def Search(norm, q, order, sigma, C0, threshold_rdp, tolerance=1e-6, max_iters=100):
    # Define the bounds for px, restricting it to (0, 1)
    lower_bound = 0.001
    upper_bound = 0.999999999

    def objective(px):
        # Objective function you want to minimize (absolute difference)
        return abs(equation(px, order, norm, sigma, C0, threshold_rdp))

    # Perform binary search
    for _ in range(max_iters):
        mid = (lower_bound + upper_bound) / 2
        f_mid = objective(mid)
        f_lower = objective(lower_bound)

        # Check if the current midpoint gives a satisfactory result
        if abs(upper_bound - lower_bound) < tolerance:
            return mid  # Found the solution within tolerance

        # Compare the values to decide which half of the interval to search next
        if f_lower < f_mid:
            upper_bound = mid  # Continue searching in the lower half
        else:
            lower_bound = mid  # Continue searching in the upper half

    # If the loop completes without converging, return the best guess (midpoint)
    return (lower_bound + upper_bound) / 2


def Fairness_sampling(rejection_sampling, train_data, clip_set, order, noise_multiplier, device, optimizer, model,
                      sample_ratio,
                      train_privacy):
    '''
    :param rejection_sampling: sampling ratio
    q_1: the sample ratio
    :alpha: the order of the rdp parameter
    :param norms: the list of the norm of the gradient
    :order: the order of the renyi differential privacy, we calculate the optimal probability
    :param sample_set: the sample_set is probability of different sample probability of different clip threshold
    :sample_ratio: the ratio of being sampled
    :return: Rejection sample and the gradient cut
    '''

    q_s = rejection_sampling
    model.train()
    clip = min(clip_set)
    threshold_rdp = compute_rdp(sample_ratio, noise_multiplier, 1, order)
    clip_sample = []
    clip_dict = {}
    # Initialize the dictionary with the minimum clip value and sample_ratio
    clip_dict[clip] = sample_ratio / q_s
    print('sample ratio:', sample_ratio / q_s)
    # for every clip we form the couple of clip and sample

    # Loop over all elements in clip_set
    for element in clip_set:
        print(element)
        if element != clip:  # Skip the minimum element
            sample = Search(element, sample_ratio, order, noise_multiplier, clip, threshold_rdp)
            # Instead of reassigning clip_dict, update it with new values
            # Create a new dictionary for the current element
            print('sample ratio:', sample / q_s)
            clip_dict[element] = sample / q_s  # Update the dictionary with new sample ratios
    fair_sample = []
    dataset = train_data.dataset
    count = 0
    tr_data_1 = Possion_sampling(train_data, device, q_s)
    tr_data = DataLoader(tr_data_1.dataset, batch_size=1)
    for id_1, (data, target) in enumerate(tr_data):  # calculate the gradient of the data
        count += 1
        data, target = data.to(device), target.to(device)
        # Zero the gradients
        optimizer.minibatch_size = len(data)
        optimizer.zero_grad()
        # Forward pass: Compute the model output
        # Compute the loss
        output = model(data)
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
        # I need to accumulate the privacy_cost in this position
        train_privacy[count - 1] + total_norm
        norm_grad, index = find_closest_value_and_index(clip_set, total_norm)
        fair_sample.append(clip_dict[norm_grad])
    random_numbers = np.random.rand(len(fair_sample))
    print('this is the total number of the data', len(fair_sample))
    # Compare random numbers with fair_sample probabilities
    successful_indices = np.where(random_numbers < fair_sample)[0]
    print(len(successful_indices))
    dataset_accept = torch.utils.data.Subset(dataset, successful_indices)
    dataset_accept_1 = DataLoader(dataset_accept, batch_size=len(successful_indices))

    return dataset_accept_1


def privacy_accumulation(test_data, model, optimizer, test_privacy):
    dataset_te = test_data.dataset  # this statement requires that test_data is a dataloader object
    te_data_loader = DataLoader(dataset_te, batch_size=1)
    count = 0
    model.train()
    for id_1, (data, target) in enumerate(te_data_loader):  # calculate the gradient of the data
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
        test_privacy[count - 1] += total_norm


def group_operation(train_privacy, test_privacy):
    train_privacy_np = np.array(train_privacy)
    test_privacy_np = np.array(test_privacy)
    # Get sorted indices based on privacy consumption
    sorted_train_indices = np.argsort(train_privacy_np)
    sorted_test_indices = np.argsort(test_privacy_np)
    # Define the group size, for example, 1000 data points per group
    train_groups = np.array_split(sorted_train_indices, len(train_privacy_np) // 20)
    test_groups = np.array_split(sorted_test_indices, len(test_privacy_np) // 20)
    # Calculate the average privacy consumption for each group
    train_group_averages = []
    test_group_averages = []
    for group in train_groups:
        average_privacy = np.mean(train_privacy_np[group])
        train_group_averages.append(average_privacy)
        print(f"Train group average privacy consumption: {average_privacy}")
    for group in test_groups:
        average_privacy = np.mean(test_privacy_np[group])
        test_group_averages.append(average_privacy)
        print(f"Test group average privacy consumption: {average_privacy}")
    return train_group_averages, test_group_averages, train_groups, test_groups


def Fairness_sampling_DPSGD_r(rejection_sampling_rate, estimated_number, start_time, train_data, test_data, model,
                            optimizer, batch_size, sample_ratio_ar, epsilon_budget, clip,
                            delta, sigma, device, fairness_parameter, training_order, MIA, file_location,
                            excel_location, round_end):
    # This step got the indices then utilize the indice.
    '''
    :MIA: is this parameter is valid, the program will return the indice of the training set will have to be returned
    :param train_data:
    :param test_data:
    :param model:
    :param optimizer:
    :param batch_size:
    :param sample_ratio:
    :param epsilon_budget:
    :param clip:
    :param delta:
    :param sigma:
    :param device:
    :param fairness_parameter: indicate how fair the training is, the larger the parameter the more unfair the training is
    :param training_order: 0 indicate the unfair training is performed first, 1 indicate the fair training is performed first
    :param MIA:  indicate whether the dataset is requiring to return, if MIA is true, we have to seperate the dataset into multiple samples, for example 1000 sample per batch
    :param file_location: file_location is the location of the out_put file to save the experiment result
    :return:
    '''
    # The minibatch size is to get the sample ratio
    sample_ratio_array = sample_ratio_ar
    sr = rejection_sampling_rate
    file = open(file_location, 'a')
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    train_size = len(train_loader.dataset)
    test_size = len(test_dl.dataset)
    orders =  list(range(2, 512)) 
    rdp_list = [0] * len(orders)
    iter = 1
    epsilon = 0.
    test_accuracy_L = []
    best_test_acc = 0.
    epsilon_list = []
    epsilon_l = []
    test_loss_list = []
    train_privacy = [0] * train_size
    test_privacy = [0] * test_size
    count = 0
    clip_un = 0.1
    rounds = []
    if isinstance(clip, (list, np.ndarray)):
        clip_un = min(clip)
    else:
        clip_un = clip
    program_start = time.monotonic()
    time_stap = []
    while epsilon < epsilon_budget:
        if count == round_end:
            break;
        if epsilon < epsilon_budget and epsilon < fairness_parameter and training_order == 0:  # unfairness training first, if the unfairness was not reachs to it's maxinmum then using the unfairness training process
            optimizer.fairness = 0
            optimizer.l2_norm_clip = clip_un
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[0]
            else:
                sample_ratio = sample_ratio_array
        elif epsilon < epsilon_budget and epsilon <= epsilon_budget - fairness_parameter and training_order == 1:  # fairness training first, if the fairness not reaches to it's maximum, then keept fair training
            optimizer.fairness = 1
            optimizer.l2_norm_clip = clip
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[1]
            else:
                sample_ratio = sample_ratio_array
        elif epsilon < epsilon_budget and epsilon >= fairness_parameter and training_order == 0:  # fair second,if the unfair reachs to it's minimum, then stop the unfair training
            optimizer.fairness = 1
            optimizer.l2_norm_clip = clip
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[1]
            else:
                sample_ratio = sample_ratio_array
        elif epsilon < epsilon_budget and epsilon >= epsilon_budget - fairness_parameter and training_order == 1:  # fair first, then no need to be fair
            optimizer.fairness = 0
            optimizer.l2_norm_clip = clip_un
            if isinstance(sample_ratio_array, (list, tuple, np.ndarray)):  # np.ndarray for NumPy arrays
                sample_ratio = sample_ratio_array[0]
            else:
                sample_ratio = sample_ratio_array
        epsilon, order, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio, sigma, rdp_list, orders,
                                                                              delta)  # comupte privacy cost
        if optimizer.fairness == 0:
            train_dl = Possion_sampling(train_loader, device, sample_ratio)  # possion sampling

        else:
            train_dl = Fairness_sampling(sr, train_loader, clip, order, sigma, device, optimizer, model,
                                         sample_ratio, train_privacy)  # possion sampling
        if MIA == 1:
            privacy_accumulation(test_dl, model, optimizer, test_privacy)
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)
        train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)
        test_loss, test_accuracy = validation(model, test_dl, device)
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter
        epsilon_list.append(torch.tensor(epsilon))
        epsilon_l.append(epsilon)
        test_accuracy_L.append(test_accuracy)
        test_loss_list.append(test_loss)
        program_end = time.monotonic()
        count = count + 1
        rounds.append(count)
        end_time = time.monotonic()
        time_elapsed = end_time - program_start
        time_stap.append(time_elapsed)
        output_string_time = (f'one step training cost {(program_end - program_start) / 3600} hours, '
                              f'so far the program has complete {(count / estimated_number) * 100}%,'
                              f'so far the program has run {(program_end - start_time) / 3600} hours, '
                              f' the program will completed in {(program_end - program_start) / 3600 * (estimated_number - count)} hours.')
        print(output_string_time)
        file.write(output_string_time)
        output_string = (
            f'iters: {iter}, '
            f'epsilon: {epsilon:.4f} | '
            f'Test set: Average loss: {test_loss:.4f}, '
            f'Accuracy: ({test_accuracy:.2f}%)'
        )
        file.write(output_string)
        print(
            f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter += 1
    print("------finished ------")
    output_string_timer = (
        f'The time to complete the program is {time_elapsed / 3600} hours.\n '
    )
    file.write(output_string_timer)
    print(output_string_timer)
    file.close()
    results = {
        'Epsilon': epsilon_l,
        'Round': rounds,
        'Time': time_stap,
        'Test Accuracy': test_accuracy_L
    }
    df = pd.DataFrame(results)
    df.to_excel(f"{excel_location}.xlsx", index=False)
    if MIA == 1:
        train_group_averages, test_group_averages, train_groups, test_groups = group_operation(train_privacy,
                                                                                               test_privacy)
        return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list,
                                                                      test_loss_list], train_group_averages, test_group_averages, train_groups, test_groups
    else:
        return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]


if __name__ == '__main__':
    dataset = 'CIFAR-10'
    momentum = 0.9
    input_norm = "BN"
    device = 'cpu'
    batch_size = 3000
    delta = 0.00001
    train_data, test_data, dataset = get_data(dataset, augment=False)
    model = get_model('DPSGD', 'CIFAR-10', 'cpu')
    optimizer = get_dp_optimizer(dataset, 'DPSGD', 0.5, momentum, [0.1, 0.11], 0.8, batch_size, model, 0, 1)
    time_start = time.time()
    test_accuracy, iter_1, best_test_acc, best_iter, model, [epsilon_list, test_loss_list] = Fairness_sampling_DPSGD_r(
        0.5,
        133, time_start,
        train_data, test_data, model, optimizer, batch_size, 0.32, 3,
        [0.1, 0.11],
        delta, 1.2, device, 0, 0, 0, './data/experiment_results.txt')
