import itertools
import torch
import numpy as np
import math
import gc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as patches

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
    fig, axes = plt.subplots(1, 1 + len(environments), gridspec_kw={'width_ratios': [20 for _ in environments] + [1]})
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


def plot_actions_over_training(network, environments, title=None):
    """
    Scatter the positions at which the greedy agent licks (action == 1) across the treadmill as a
    function of training step, one panel per treadmill. Reward locations of every treadmill are shaded
    so it is clear whether licking is appropriately gated by context.

    :param network: trained Q_Network with a populated val_action_history (set during network.train)
    :param environments: list of Cue_Treadmill environments
    :param title: optional str appended to the figure title
    """
    if not hasattr(network, 'val_action_history') or len(network.val_action_history) == 0:
        print('plot_actions_over_training: network has no val_action_history; skipping')
        return

    action_history = np.asarray(network.val_action_history)  # [n_val_steps, n_environments, length]
    train_steps = np.asarray(network.val_history_train_steps)

    n_env = len(environments)
    fig, axes = plt.subplots(1, n_env, figsize=(4 * n_env, 4), squeeze=False)
    axes = axes[0]

    colors = ['r', 'b', 'y', 'g', 'm', 'c'][:n_env]
    legend_elements = [patches.Patch(facecolor=color, alpha=0.2, label=f'Treadmill {j + 1} reward')
                       for j, color in enumerate(colors)]

    for i, environment in enumerate(environments):
        ax = axes[i]
        # shade the reward location of every treadmill for context
        for environment_j, color in zip(environments, colors):
            ax.axvspan(environment_j.reward_position - 1, environment_j.reward_position + 1,
                       alpha=0.2, color=color, ls='')
        # cue location
        ax.axvline(environment.cue_position, color='gray', linestyle='--', linewidth=1)

        lick_idx = np.argwhere(action_history[:, i, :] > 0)  # (val_step_index, position)
        if lick_idx.size > 0:
            ax.scatter(lick_idx[:, 1], train_steps[lick_idx[:, 0]], marker='o', c='g', s=4)

        ax.invert_yaxis()
        ax.set_xlabel('Position')
        ax.set_title('Treadmill {}'.format(i + 1))
        ax.set_xlim(-0.5, environment.length - 0.5)
        if i == 0:
            ax.set_ylabel('Training step')
            ax.legend(handles=legend_elements, loc='lower left')

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)
    fig.suptitle('Agent licks over training%s' % title_str)
    fig.tight_layout()
    plt.show(block=False)


def plot_hidden_state_cross_correlation(network, environments, population_name=None, title=None):
    """
    Plot the cross-correlation between hidden-population representations of the trained network across
    pairs of treadmills. For each pair (i, j), entry [a, b] of the heatmap is the correlation (across
    hidden units) between the activity vector at position a of treadmill i and position b of treadmill j.

    :param network: trained Q_Network
    :param environments: list of Cue_Treadmill environments
    :param population_name: optional str fullname of hidden population (e.g. 'H1E'). Defaults to the
        first population of the layer feeding the output layer.
    :param title: optional str appended to the figure title
    """
    activities = network.get_treadmill_hidden_activity(environments, population_name=population_name)

    n_env = len(environments)
    pairs = [(i, j) for i in range(n_env) for j in range(i + 1, n_env)]
    if len(pairs) == 0:
        print('plot_hidden_state_cross_correlation: need at least two environments; skipping')
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4), squeeze=False)
    axes = axes[0]

    im = None
    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        t1 = activities[i]  # [length_i, n_units]
        t2 = activities[j]  # [length_j, n_units]
        length_i = t1.shape[0]
        # rows = treadmill i positions, cols = treadmill j positions
        corr = np.corrcoef(t1, t2)[:length_i, length_i:]

        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, origin='upper')

        env_i, env_j = environments[i], environments[j]
        # cue location (dashed gray) along both axes
        ax.axhline(env_i.cue_position, color='gray', linestyle='--', linewidth=0.75)
        ax.axvline(env_j.cue_position, color='gray', linestyle='--', linewidth=0.75)
        # reward locations (dashed red): row for treadmill i, col for treadmill j
        ax.axhline(env_i.reward_position, color='r', linestyle='--', linewidth=0.75)
        ax.axvline(env_j.reward_position, color='r', linestyle='--', linewidth=0.75)

        ax.set_xlabel('Treadmill {} position'.format(j + 1))
        ax.set_ylabel('Treadmill {} position'.format(i + 1))
        ax.set_title('Treadmill {} vs {}'.format(i + 1, j + 1))

    if im is not None:
        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.04, pad=0.04)

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)
    fig.suptitle('Cross-correlation of hidden states%s' % title_str)
    plt.show(block=False)

