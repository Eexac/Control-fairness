import time

import torch
from torch.utils.data import DataLoader

from privacy_account.comupute_dp_sgd import find_best_order
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
from scipy.optimize import minimize, minimize_scalar
import numpy as np
from privacy_account import compute_eps
from scipy.optimize import root


def constraint(sigma, px, alpha, g_norm, C0, delta, m):
    """Ensure DP constraint is met: epsilon ≤ m"""
    return compute_eps(alpha, compute_rdp(px, C0 * sigma / g_norm, 1, alpha), delta)[0] - m


def objective_sigma(sigma, norm, q, order, C0, delta, m):
    """Objective function: minimize the absolute deviation from DP constraint"""
    px = max(q, 1e-6)  # Ensure px is within valid range

    # Compute absolute privacy constraint deviation
    return abs(constraint(sigma, px, order, norm, C0, delta, m))


def find_best_sigma(norm, q, order, C0, delta, m, sigma_bounds=(0, 5.67)):
    """Search for the best sigma within the given bounds [0, 5.67]"""
    result = minimize_scalar(objective_sigma, bounds=sigma_bounds, method='bounded',
                             args=(norm, q, order, C0, delta, m))
    return result.x  # Return best found sigma


def find_best_sigma_Post(norm, q, order, C0, delta, rdp, m, sigma_bounds=(0, 5.67)):
    """Search for the best sigma within the given bounds [0, 5.67]"""
    result = minimize_scalar(objective_sigma_Post, bounds=sigma_bounds, method='bounded',
                             args=(norm, q, order, C0, delta, rdp, m))
    return result.x  # Return best found sigma


def objective_sigma_Post(sigma, norm, q, order, C0, delta, rdp, m):
    """Objective function: minimize the absolute deviation from DP constraint"""
    px = max(q, 1e-6)  # Ensure px is within a valid range
    return abs(constraint_Post(sigma, px, order, norm, C0, delta, rdp, m))  # Minimize absolute deviation


def constraint_Post(sigma, px, alpha, g_norm, C0, delta, rdp, m):
    """Ensure DP constraint is met: ε ≤ m"""
    epsilon, _, _ = apply_dp_sgd_analysis_epsilon_accumulation(px, sigma, rdp, alpha, delta)
    return epsilon - m  # Constraint function for DP bound


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


def Fairness_sampling(train_data, clip_set, clip_standard, order, noise_multiplier, device, optimizer, model,
                      sample_ratio):
    '''
    q_1: the sample ratio
    :alpha: the order of the rdp parameter
    :param norms: the list of the norm of the gradient
    :order: the order of the renyi differential privacy, we calculate the optimal probability
    :param sample_set: the sample_set is probability of different sample probability of different clip threshold
    :sample_ratio: the ratio of being sampled
    :return: Rejection sample and the gradient cut
    '''
    model.train()
    if clip_standard == 0:
        clip = min(clip_set)
    else:
        clip = clip_standard
    threshold_rdp = compute_rdp(sample_ratio, noise_multiplier, 1, order)
    clip_sample = []
    clip_dict = {}
    # Initialize the dictionary with the minimum clip value and sample_ratio
    clip_dict[clip] = sample_ratio
    print('sample ratio:', sample_ratio)
    # for every clip we form the couple of clip and sample

    # Loop over all elements in clip_set
    for element in clip_set:
        print(element)
        if element != clip:  # Skip the minimum element
            sample = Search(element, sample_ratio, order, noise_multiplier, clip, threshold_rdp)
            # Instead of reassigning clip_dict, update it with new values
            # Create a new dictionary for the current element
            print('sample ratio:', sample)
            clip_dict[element] = sample  # Update the dictionary with new sample ratios
    fair_sample = []
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


def Fairness_sampling_DPSGD(estimated_number, start_time, train_data, test_data, model, optimizer, batch_size,
                            sample_ratio_ar, epsilon_budget, clip, clip_standard,
                            delta, sigma, device, fairness_parameter, training_order, rdp_order, one_shot,
                            file_location):
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
    file = open(file_location, 'a')
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    train_size = len(train_loader.dataset)
    test_size = len(test_dl.dataset)
    order = rdp_order
    rdp_list = [0] * len(order)
    iter = 1
    epsilon = 0.
    best_test_acc = 0.
    epsilon_list = []
    test_loss_list = []
    test_accuracy = 0
    count = 0
    order1 = 0
    best_iter = 1
    if one_shot == 1 and fairness_parameter > 0:
        program_start = time.time()
        if training_order == 0:
            iter = 0
            while epsilon < fairness_parameter:
                epsilon, order1, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_array[0], sigma,
                                                                                       rdp_list,
                                                                                       order, delta)
                train_dl = Possion_sampling(train_loader, device, sample_ratio_array[0])  # possion sampling
                optimizer.minibatch_size = batch_size[0]
                train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)
                test_loss, test_accuracy = validation(model, test_dl, device)
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    best_iter = iter
                test_loss_list.append(test_loss)
                program_end = time.time()
                count = count + 1
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
            optimizer.minibatch_size = batch_size[1]
            sigma_one_shot = find_best_sigma_Post(1, sample_ratio_array[1], order, 1, delta, rdp_list, epsilon_budget,
                                                  sigma_bounds=(0, 5.67))
            print(sigma_one_shot)
            optimizer.lr = 0.00000000001
            optimizer.sigma = sigma_one_shot
            train_dl = Fairness_sampling(train_loader, clip, clip_standard, order, sigma_one_shot, device, optimizer,
                                         model,
                                         sample_ratio_array[1])
            train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, device)
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_iter = iter
            test_loss_list.append(test_loss)
            program_end = time.time()
            count = count + 1
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

            print("------finished ------")
            end_time = time.time()
            output_string_timer = (
                f'The time to complete the program is {(start_time - end_time) / 3600} hours.\n '
            )
            file.write(output_string_timer)
            print(output_string_timer)
            file.close()
            return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]
        else:
            iter = 1
            sigma_one_shot = find_best_sigma(1, sample_ratio_array[1], order, 1, delta,
                                             epsilon_budget - fairness_parameter,
                                             sigma_bounds=(0, 5.67))
            optimizer.lr = 0.00000000001
            optimizer.sigma = sigma_one_shot
            train_dl = Fairness_sampling(train_loader, clip, clip_standard, order, sigma_one_shot, device, optimizer,
                                         model,
                                         sample_ratio_array[1])
            train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, device)
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_iter = iter
            test_loss_list.append(test_loss)
            program_end = time.time()
            count = count + 1
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

            print("------finished ------")
            end_time = time.time()
            output_string_timer = (
                f'The time to complete the program is {(start_time - end_time) / 3600} hours.\n '
            )
            file.write(output_string_timer)
            print(output_string_timer)
            print(sigma_one_shot)
            epsilon, order1, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_array[1],
                                                                                   sigma_one_shot,
                                                                                   rdp_list,
                                                                                   order, delta)
            print(epsilon)
            while epsilon < epsilon_budget:
                epsilon, order1, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio_array[0], sigma,
                                                                                       rdp_list,
                                                                                       order, delta)
                train_dl = Possion_sampling(train_loader, device, sample_ratio_array[0])  # possion sampling
                optimizer.minibatch_size = batch_size[0]
                train_loss, train_accuracy = train_with_dp(model, train_dl, optimizer, device)
                test_loss, test_accuracy = validation(model, test_dl, device)
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    best_iter = iter
                test_loss_list.append(test_loss)
                program_end = time.time()
                count = count + 1
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
            end_time = time.time()
            output_string_timer = (
                f'The time to complete the program is {(start_time - end_time) / 3600} hours.\n '
            )
            file.write(output_string_timer)
            print(output_string_timer)
            file.close()
            return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]

    while epsilon < epsilon_budget:
        program_start = time.time()
        sample_ratio = 0
        if epsilon < epsilon_budget and epsilon < fairness_parameter and training_order == 0:  # unfairness training first, if the unfairness was not reachs to it's maxinmum then using the unfairness training process
            optimizer.fairness = 0
            optimizer.minibatch_size = batch_size[0]
            sample_ratio = sample_ratio_array[0]
            optimizer.l2_norm_clip = clip_standard
        elif epsilon < epsilon_budget and epsilon <= epsilon_budget - fairness_parameter and training_order == 1:  # fairness training first, if the fairness not reaches to it's maximum, then keept fair training
            optimizer.fairness = 1
            optimizer.minibatch_size = batch_size[1]
            sample_ratio = sample_ratio_array[0]
            optimizer.l2_norm_clip = clip
        elif epsilon < epsilon_budget and epsilon >= fairness_parameter and training_order == 0:  # fair second,if the unfair reachs to it's minimum, then stop the unfair training
            optimizer.fairness = 1
            optimizer.minibatch_size = batch_size[1]
            sample_ratio = sample_ratio_array[1]
            optimizer.l2_norm_clip = clip
        elif epsilon < epsilon_budget and epsilon >= epsilon_budget - fairness_parameter and training_order == 1:  # fair first, then no need to be fair
            optimizer.fairness = 0
            optimizer.minibatch_size = batch_size[0]
            sample_ratio = sample_ratio_array[0]
            optimizer.l2_norm_clip = clip_standard
        epsilon, order1, rdp_list = apply_dp_sgd_analysis_epsilon_accumulation(sample_ratio, sigma, rdp_list, order,
                                                                               delta)  # comupte privacy cost
        if optimizer.fairness == 0:
            train_dl = Possion_sampling(train_loader, device, sample_ratio)  # possion sampling

        else:
            train_dl = Fairness_sampling(train_loader, clip, clip_standard, order, sigma, device, optimizer, model,
                                         sample_ratio)  # possion sampling
        for id, (data, target) in enumerate(train_dl):
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
    end_time = time.time()
    output_string_timer = (
        f'The time to complete the program is {(start_time - end_time) / 3600} hours.\n '
    )
    file.write(output_string_timer)
    print(output_string_timer)
    file.close()
    return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]


if __name__ == '__main__':
    one_shot = 1
    orders = list(range(2, 64)) + [128, 256, 512]
    # datasets  has four options 'CIFAR-10', 'MNIST', 'IMDB', 'FMNIST'
    training_order = 0  # 0 represent unfairness training first, 1 represent fairness training first
    # this is the example of CIFAR-10
    dataset = 'CIFAR-10'
    sampling_ratio_unfair = 0
    sampling_ratio_fair = 0
    if dataset == 'IMDB' or dataset == 'CIFAR-10':
        sampling_ratio_unfair = 8192 / 50000
        sampling_ratio_fair = 19200 / 50000
    else:
        sampling_ratio_unfair = 8192 / 60000
        sampling_ratio_fair = 19200 / 60000
    order = find_best_order(1, orders, 0.5, 3, 0.00001, 5.67, sampling_ratio_fair, sampling_ratio_unfair)
    momentum = 0.9
    device = 'cpu'
    batch_size_fairness = 19200
    batch_size_unfairness = 8192
    delta = 0.00001
    train_data, test_data, dataset = get_data(dataset, augment=False)
    model = get_model('DPSGD', 'CIFAR-10', 'cpu')
    optimizer = get_dp_optimizer(dataset, 'DPSGD', 0.5, momentum, 0.1, 5.67, 8192, model, 0, 1)
    optimizer.minibatch_size = 1
    test_accuracy, iter_1, best_test_acc, best_iter, model, [epsilon_list, test_loss_list] = Fairness_sampling_DPSGD(
        133,
        time.time(),
        train_data,
        test_data,
        model,
        optimizer,
        [batch_size_unfairness, batch_size_fairness],
        [sampling_ratio_unfair, sampling_ratio_fair],
        3, [0.1, 0.11, 0.12], 0.1, delta,
        5.67,
        device,
        0.5, 1,
        order, one_shot,
        'data/experiment_DP-SGD_CIFAR.txt')
