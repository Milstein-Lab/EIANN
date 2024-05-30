"""
This module contains various learning rules for updating weights in neural networks.
"""

from .base_classes import LearningRule, BiasLearningRule
from .backprop import Backprop, BackpropBias, Backprop_EWC, Backprop_DendriticLoss
from .backprop_like import *
from .hebbian import *
from .btsp import *
from .dendritic_loss import *
from .weight_functions import *

# __all__ = [
#     'LearningRule', 'BiasLearningRule',
#     'Backprop', 'BackpropBias', 'Backprop_EWC', 'Backprop_DendriticLoss',
#     'HebbianRule', 'OjaRule', 'Hebb_WeightNorm',
#     'DendriticLoss', 'DendriticLossBias',
#     'clone_weight', 'normalize_weight', 'no_autapses',
# ]
