"""
Activation functions.
"""

import torch
import numpy as np
import EIANN.external as external



def set_activation(network, activation, **kwargs):
    # Set callable activation function
    if isinstance(activation, str):
        if activation in globals():
            activation = globals()[activation]
        elif hasattr(external, activation):
            activation = getattr(external, activation)
    if not callable(activation):
        raise RuntimeError \
            ('Population: callable for activation: %s must be imported' % activation)
    activation_f = lambda x: activation(x, **kwargs)

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                population.activation = activation_f


def srelu(x, min=0, max=1):
    return torch.clamp(x, min, max)


def linear(x):
    """
    Linear activation function
    """
    return x


def softmax(x):
    """
    Softmax activation function
    """
    return torch.nn.functional.softmax(x, dim=-1)


def get_scaled_rectified_sigmoid_orig(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError('scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return np.vectorize(
        lambda xi:
        (target_amp / amp) * (1. / (1. + np.exp(-slope * (max(min(xi, x[-1]), x[0]) - th))) - start_val) + ylim[0])


def get_scaled_rectified_sigmoid(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError('scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return lambda xi: (target_amp / amp) * (1. / (1. + torch.exp(-slope * (torch.clamp(xi, x[0], x[-1]) - th))) -
                                            start_val) + ylim[0]

