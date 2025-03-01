import math
from privacy_account.compute_rdp import compute_rdp, compute_rdp_randomized_response, compute_rdp_accumulation
from privacy_account.rdp_convert_dp import compute_eps
import numpy as np


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters.
    Args:
      n: Number of examples in the training data.
      batch_size: Batch size used in training.
      noise_multiplier: Noise multiplier used in training.
      epochs: Number of epochs in training.
      delta: Value of delta for which to compute epsilon.
      S:sensitivity
    Returns:
      Value of epsilon corresponding to input hyperparameters.
    """
    q = batch_size / n
    if q > 1:
        print('n must be larger than the batch size.')
    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda

    steps = int(math.ceil(epochs * (n / batch_size)))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    rdp = compute_rdp(q, sigma, steps, orders)

    eps, opt_order = compute_eps(orders, rdp, delta)

    return eps, opt_order, rdp


def apply_dp_sgd_analysis_epsilon(q, sigma, steps, orders, epsilon):
    """Compute and print results of DP-SGD analysis."""

    rdp = compute_rdp(q, sigma, steps, orders)
    from privacy_account import compute_delta
    delta, opt_order_1 = compute_delta(orders, rdp, epsilon)

    return delta, opt_order_1


def apply_dp_sgd_analysis_epsilon_accumulation(q, sigma, privacy_before, orders, delta):
    """Compute and print results of DP-SGD analysis."""
    rdp = compute_rdp_accumulation(q, sigma, privacy_before, orders)
    eps, opt_order = compute_eps(orders, rdp, delta)

    return eps, opt_order, rdp


def RR_dp_privacy(p, steps, delta):
    orders = (list(range(2, 64)) + [128, 256, 512])

    rdp = compute_rdp_randomized_response(p, steps, orders)

    eps, opt_order = compute_eps(orders, rdp, delta)
    return eps, opt_order


def find_closest_indices(epsilon_array, target_epsilons):
    closest_indices = []
    for target in target_epsilons:
        # Calculate absolute differences between target and all elements in epsilon_array
        differences = [abs(target - epsilon) for epsilon in epsilon_array]
        # Find the index of the smallest difference
        closest_index = differences.index(min(differences))
        closest_indices.append(closest_index + 1)
    return closest_indices


def find_best_order(traning_order,orders, Phi, epsilon, delta, sigma, sampling_ratio_fairness,sampling_ratio_unfairness):

    epsilon_curent = 0
    best_alpha = 0
    count = 0
    privacy_before = [0]*len(orders)
    order = 0
    if traning_order == 0:
        while epsilon_curent < Phi:
            count += 1
            epsilon_curent, order, privacy_before = apply_dp_sgd_analysis_epsilon_accumulation(sampling_ratio_unfairness, sigma, privacy_before, orders, delta)
        count_1 = 0
        while epsilon_curent < 3:
            count += 1
            epsilon_curent, best_alpha, privacy_before = apply_dp_sgd_analysis_epsilon_accumulation(sampling_ratio_unfairness, sigma,
                                                                                             privacy_before,
                                                                                             orders,
                                                                                             delta)  # comupte privacy cost
    else:
        while epsilon_curent < epsilon - Phi:
            count += 1
            epsilon_curent, order, privacy_before = apply_dp_sgd_analysis_epsilon_accumulation(sampling_ratio_fairness, sigma, privacy_before, orders, delta)
        count_1 = 0
        while epsilon_curent < 3:
            count += 1
            epsilon_curent, best_alpha, privacy_before = apply_dp_sgd_analysis_epsilon_accumulation(sampling_ratio_unfairness,  sigma,
                                                                                privacy_before,
                                                                                             orders,
                                                                                             delta)  # comupte privacy cost
    return [best_alpha]

#fairness traning first


if __name__ == "__main__":
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    privacy_before = [0] * len(orders)
    epsilon = 0
    count = 1
    delta = 0.00001
    epsilon_array = []
    Phi = 0.5
    sampling_ratio = [0.1, 0.6]

    print(find_best_order(1, orders, 0.7, 3, delta, 5.67,0.136, 0.136,
                    ))
    # closest_indices = find_closest_indices(epsilon_array, epsilons)
    # print("Closest indices:", closest_indices)

    # for i in range(800):
    # epsilon, order, privacy_before = apply_dp_sgd_analysis_epsilon_accumulation(0.136, 5.67, privacy_before, orders, 0.00001)
    # print('privacy:', epsilon, 'order:', order, i )
