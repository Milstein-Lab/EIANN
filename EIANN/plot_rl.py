import itertools
import torch
import numpy as np
import math
import gc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import scipy.stats as stats

from tqdm.autonotebook import tqdm
import copy
import EIANN.utils as ut


def update_plot_defaults():
    font_size = 7
    plt.rcParams.update({"font.size": font_size,
                    "figure.titlesize": font_size,
                    "figure.labelweight": font_size,
                    "axes.titlesize": font_size,
                    "axes.labelsize": font_size,
                    "xtick.labelsize": font_size,
                    "ytick.labelsize": font_size,
                    "legend.fontsize": font_size,
                    "legend.title_fontsize": font_size,                    
                    "axes.spines.right": False,
                    "axes.spines.top": False,
                    "axes.linewidth": 0.5,
                    "lines.linewidth": 0.5,
                    "lines.markersize": 3,
                    "xtick.major.size": 2.5,
                    "ytick.major.size": 2.5,
                    "xtick.minor.size": 2,
                    "ytick.minor.size": 2,
                    "xtick.minor.width": 0.5,
                    "ytick.minor.width": 0.5,
                    "xtick.major.width": 0.5,
                    "ytick.major.width": 0.5,
                    "xtick.major.pad":   2,
                    "ytick.major.pad":   2,    
                    "xtick.minor.pad":   2,
                    "ytick.minor.pad":   2,
                    "legend.frameon": False,
                    "savefig.transparent": True,
                    "legend.handletextpad": 0.5,
                    "legend.handlelength": 1.,
                    "legend.labelspacing": 0.3,
                    "legend.columnspacing": 1.2,
                    "figure.figsize": [4, 1.5],
                    "figure.dpi": 200,
                    "font.sans-serif": 'Avenir',
                    "text.usetex": False,
                    "svg.fonttype": 'none',
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42})


def clean_axes(axes, left=True, right=False):
    """
    Remove top and right axes from pyplot axes object.
    :param axes: list of pyplot.Axes
    :param top: bool
    :param left: bool
    :param right: bool
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        if not right:
            axis.spines['right'].set_visible(False)
        if not left:
            axis.spines['left'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()


# *******************************************************************
# Network summary functions
# *******************************************************************
def plot_validation_rewards(network, title=None, train_step_range=None, ax=None):
    assert len(network.val_reward_history) > 0, 'Network must contain a stored val_loss_history'

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)

    if train_step_range is None:
        train_steps = network.val_history_train_steps
        train_step_range = (network.val_history_train_steps[0], network.val_history_train_steps[-1])
        val_reward_history = network.val_reward_history
    else:
        train_steps_idx = np.where((network.val_history_train_steps >= train_step_range[0]) & \
                                    (network.val_history_train_steps <= train_step_range[1]))[0]
        train_steps = network.val_history_train_steps[train_steps_idx]
        val_reward_history = network.val_reward_history[train_steps_idx]

    if ax is None:
        fig = plt.figure()
        plt.plot(train_steps, val_reward_history, linewidth=1)
        plt.axhline(1, linestyle='--', color='gray')
        plt.xlabel('Training steps')
        plt.ylabel('Average reward')
        # plt.xlim(train_step_range[0], train_step_range[1])
        fig.suptitle('Validation rewards%s' % title_str)
        fig.tight_layout()
        plt.show(block=False)
    else: 
        ax.plot(train_steps, val_reward_history, label='Validation rewards', color='r', linewidth=1)
        ax.set_xlabel('Training steps')


def plot_final_q_vals(network, environments):
    fig, axes = plt.subplots(1, 1 + len(environments), gridspec_kw={'width_ratios': [20, 20, 1]})
    _, final_q_vals = network.test(environments, return_q_vals=True)

    vmin = final_q_vals.min()
    vmax = final_q_vals.max()

    for i, (tread, q_vals) in enumerate(zip(environments, final_q_vals)):
            
        im = axes[i].imshow(q_vals, cmap='plasma', vmin=vmin, vmax=vmax, )
        axes[i].set_xlabel('Action')
        axes[i].set_ylabel("cue position: {}, cue value: {}, reward position: {}".format(tread.cue_position, tread.cue_number, tread.reward_position))
        axes[i].set_xticks([0, 1], ['I', 'L'])
        axes[i].set_yticks(list(range(tread.length)))

        axes[i].set_title('Treadmill {}'.format(i+1))

    cbar = plt.colorbar(im, cax=axes[-1])
    fig.suptitle('Final Q Values', fontsize=16)

