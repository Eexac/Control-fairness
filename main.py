import argparse
import os
import pickle
from datetime import time, datetime
import pandas as pd
import torch
import time
from Fairness_Algorithms import Fairness_US_DPSGD
from data.util.get_data import get_data
from Fairness_Algorithms.DPSGD import DPSGD
from Fairness_Algorithms.Individual_privacy_accounting import Fairness_DPSGD_US_Accounting
from Fairness_Algorithms.Fake_fairness_DP_SGD import Fake_fairness_DPSGD
from Fairness_Algorithms.Fairness_sampling_DP_SGD import Fairness_sampling_DPSGD
from data.util.get_data import get_data
from model.get_model import get_model
from privacy_account import find_best_order
from utils.dp_optimizer import get_dp_optimizer
import ast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="DPSGD",
                        choices=['DPSGD', 'DPSGD-US', 'DPSGD-FS', 'FAKE-DPSGD'])
    parser.add_argument('--dataset_name', type=str, default="MNIST", choices=['MNIST', 'FMNIST', 'CIFAR-10', 'IMDB'])
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--fairness_parameter', type=float, default=1)
    parser.add_argument('--sigma_t', type=float, default=5.67)
    parser.add_argument('--C_t', type=float, default=0.1)
    parser.add_argument('--clip_set', type=str, default="[0.1, 0.11, 0.12]")
    parser.add_argument('--epsilon', type=float, default=3.0)
    parser.add_argument('--training_order', type=float, default=1, choices=[0, 1])
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--batch_size_fair', type=int, default=19200)
    parser.add_argument('--one_shot', type=float, default=1)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    one_shot = args.one_shot
    training_order = args.training_order
    algorithm = args.algorithm
    dataset_name = args.dataset_name
    lr = args.lr
    momentum = args.momentum
    sigma = args.sigma_t
    C_t = args.C_t
    args.clip_set = ast.literal_eval(args.clip_set)
    clip_set = args.clip_set
    epsilon = args.epsilon
    delta = args.delta
    batch_size = args.batch_size
    batch_size_fair = args.batch_size_fair
    Phi = args.fairness_parameter
    device = args.device
    train_data, test_data, dataset = get_data(dataset_name, augment=False)
    model = get_model(algorithm, dataset_name, device)
    optimizer = get_dp_optimizer(dataset_name, algorithm, lr, momentum, C_t, sigma, batch_size, model, 0, 1)
    batch_size_array = [batch_size, batch_size_fair]
    sample_ratio_unfairness = 0
    sample_ratio_fairness = 0
    orders = list(range(2, 512))
    start_time = time.time()
    if dataset_name == 'IMDB' or dataset_name == 'CIFAR-10':
        sample_ratio_unfairness = batch_size / 50000
        sample_ratio_fairness = batch_size_fair / 50000
    else:
        sample_ratio_unfairness = batch_size / 60000
        sample_ratio_fairness = batch_size_fair / 60000

    if algorithm == 'DPSGD':
        test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD(
            train_data, test_data, model, optimizer, batch_size,
            epsilon, delta, sigma, device
        )

    elif algorithm == 'DPSGD-FS':
        order = find_best_order(
            training_order, orders, Phi, epsilon, delta, sigma,
            sample_ratio_fairness, sample_ratio_unfairness
        )

        test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = Fairness_sampling_DPSGD(
            130, start_time, train_data, test_data, model, optimizer, batch_size_array,
            [sample_ratio_unfairness, sample_ratio_fairness], epsilon, clip_set, C_t,
            delta, sigma, device, Phi, training_order, order, one_shot,
            f'data/experiment_DP-SGD_FS_{dataset_name}.txt'
        )

    elif algorithm == 'DPSGD-US' and dataset_name != 'IMDB':  # Not supported for IMDB
        order = find_best_order(
            training_order, orders, Phi, epsilon, delta, sigma,
            sample_ratio_fairness, sample_ratio_unfairness
        )

        test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = Fairness_US_DPSGD(
            130, start_time, train_data, test_data, model, optimizer, batch_size,
            [sample_ratio_unfairness, sample_ratio_fairness], epsilon, C_t,
            delta, sigma, device, Phi, training_order, order,
            f'data/experiment_DP-SGD_US_{dataset_name}.txt'
        )

    elif algorithm == "FAKE-DPSGD":
        order = find_best_order(
            training_order, orders, Phi, epsilon, delta, sigma,
            sample_ratio_fairness, sample_ratio_unfairness
        )

        test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = Fake_fairness_DPSGD(
            130, time.time(), train_data, test_data, model, optimizer, batch_size,
            sample_ratio_unfairness, epsilon, C_t, delta, sigma, device, order,
            f'data/experiment_FAKE_{dataset_name}.txt'
        )

    else:
        raise ValueError("This algorithm does not exist.")


if __name__ == "__main__":
    print("Debug: main() function started")  # Add this line
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    main()
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start time: ", start_time)
    print("end time: ", end_time)
