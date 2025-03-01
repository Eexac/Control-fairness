import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop


def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, fairness, random, *args,
                     **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.fairness = fairness
            self.random = random
            # random is used to indicate whether the scale of the gradient was random
            for id, group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in
                                        group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def get_closest_value_and_index(self, total_norm):
            # Convert to a NumPy array if it's not already
            l2_norm_clip = np.array(self.l2_norm_clip)

            # Find the index of the closest value in l2_norm_clip to total_norm
            index = np.abs(l2_norm_clip - total_norm).argmin()

            # Get the closest value
            closest_value = l2_norm_clip[index]

            return closest_value

        def microbatch_step(self):  # usually the microbatch is setted to 1
            clip = 0
            if np.isscalar(self.l2_norm_clip) and self.fairness == 0:
                clip = self.l2_norm_clip
                total_norm = 0.
                for group in self.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            total_norm += param.grad.data.norm(2).item() ** 2.

                total_norm = total_norm ** .5
                clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

                for group in self.param_groups:
                    for param, accum_grad in zip(group['params'], group['accum_grads']):
                        if param.requires_grad:
                            accum_grad.add_(param.grad.data.mul(clip_coef))

                return min(total_norm, self.l2_norm_clip)
            if np.isscalar(
                    self.l2_norm_clip) and self.fairness == 1 and self.random == 0:  # the scale was the same, that every
                total_norm = 0.
                clip = self.l2_norm_clip
                for group in self.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            total_norm += param.grad.data.norm(2).item() ** 2.

                total_norm = total_norm ** .5
                if total_norm == 0.0:
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                if param.grad.data.eq(0).all():  # Check if the gradient is all zeros
                                    # Create a tensor with all elements set to some value (e.g., 1.0)
                                    pad_value = torch.full_like(param.grad.data, 1.0)
                                    # Normalize the padded tensor to have an L2 norm of clip_coef
                                    norm_pad_value = pad_value / pad_value.norm(
                                        p=2) * self.l2_norm_clip  # Normalize to clip_coef norm
                                    accum_grad.add_(norm_pad_value)
                else:
                    clip_coef = self.l2_norm_clip / total_norm
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                accum_grad.add_(param.grad.data.mul(clip_coef))
                return self.l2_norm_clip
            if np.isscalar(
                    self.l2_norm_clip) and self.fairness == 1 and self.random == 1:  # the scale was the same, that every
                total_norm = 0.
                clip = self.l2_norm_clip
                for group in self.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            total_norm += param.grad.data.norm(2).item() ** 2.

                total_norm = total_norm ** .5
                if total_norm == 0.0:
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                # Create a tensor with random values between 0 and 1
                                pad_value = torch.rand_like(param.grad.data)
                                # Normalize the padded tensor to have an L2 norm of clip
                                norm_pad_value = pad_value / pad_value.norm(
                                    p=2) * self.l2_norm_clip  # Normalize to clip norm
                                accum_grad.add_(norm_pad_value)
                else:
                    clip_coef = self.l2_norm_clip / total_norm
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                accum_grad.add_(param.grad.data.mul(clip_coef))
                return self.l2_norm_clip
            elif (isinstance(self.l2_norm_clip, np.ndarray) or isinstance(self.l2_norm_clip,
                                                                          list)) and self.fairness == 1 and self.random == 0:
                total_norm = 0.
                for group in self.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            total_norm += param.grad.data.norm(2).item() ** 2.
                total_norm = total_norm ** .5
                clip = self.get_closest_value_and_index(total_norm)
                if total_norm == 0.0:
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                if param.grad.data.eq(0).all():  # Check if the gradient is all zeros
                                    # Create a tensor with all elements set to some value (e.g., 1.0)
                                    pad_value = torch.full_like(param.grad.data, 1.0)
                                    # Normalize the padded tensor to have an L2 norm of clip_coef
                                    norm_pad_value = pad_value / pad_value.norm(
                                        p=2) * clip  # Normalize to clip_coef norm
                                    accum_grad.add_(norm_pad_value)
                else:
                    clip_coef = clip / total_norm
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                accum_grad.add_(param.grad.data.mul(clip_coef))
            elif (isinstance(self.l2_norm_clip, np.ndarray) or isinstance(self.l2_norm_clip,
                                                                          list)) and self.fairness == 1 and self.random == 1:
                total_norm = 0.
                for group in self.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            total_norm += param.grad.data.norm(2).item() ** 2.
                total_norm = total_norm ** .5
                clip = self.get_closest_value_and_index(total_norm)
                if total_norm == 0.0:
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                # Create a tensor with random values between 0 and 1
                                pad_value = torch.rand_like(param.grad.data)
                                # Normalize the padded tensor to have an L2 norm of clip
                                norm_pad_value = pad_value / pad_value.norm(p=2) * clip  # Normalize to clip norm
                                accum_grad.add_(norm_pad_value)
                else:
                    clip_coef = clip / total_norm
                    for group in self.param_groups:
                        for param, accum_grad in zip(group['params'], group['accum_grads']):
                            if param.requires_grad:
                                accum_grad.add_(param.grad.data.mul(clip_coef))

            return clip

        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step_dp(self, *args,
                    **kwargs):  # this is the average step, in this step the accumulated gradient will be averaged
            clip = 0
            if isinstance(self.l2_norm_clip, (np.ndarray, list)):
                clip = min(self.l2_norm_clip)
            else:
                clip = self.l2_norm_clip
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(
                            clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
                        # This ensures that after accumulating the gradients from all microbatches, the final gradient update is scaled down by the ratio of microbatch_size / minibatch_size. This effectively averages the gradients over the number of microbatches (which form the minibatch).
            super(DPOptimizerClass, self).step(*args, **kwargs)

        def step_dp_agd(self, *args, **kwargs):
            clip = 0
            if isinstance(self.l2_norm_clip, (np.ndarray, list)):
                clip = min(self.l2_norm_clip)
            else:
                clip = self.l2_norm_clip
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(
                            clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

    return DPOptimizerClass


DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)


def get_dp_optimizer(dataset_name, algortithm, lr, momentum, C_t, sigma, batch_size, model, fairness, random):
    if dataset_name == 'IMDB' and algortithm != 'DPAGD':
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
            fairness=fairness,
            random=random
        )
    else:
        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
            fairness=fairness,
            random=random
        )
    return optimizer
