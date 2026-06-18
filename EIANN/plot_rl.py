import itertools
import os
import torch
import numpy as np
import math
import gc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as patches
import seaborn as sns  # registers the 'icefire' (and 'rocket', 'mako', ...) colormaps with matplotlib

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


def _save_or_show(fig, save_path, dpi=600):
    """
    Save the figure to save_path if provided (creating parent directories as needed) and close it;
    otherwise display it without blocking.

    :param fig: matplotlib Figure
    :param save_path: str or None
    :param dpi: int; resolution for the saved raster figure (publication-quality by default)
    """
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show(block=False)


# *******************************************************************
# Network summary functions
# *******************************************************************
def _prefix_title(title, base):
    """Prefix the (optional) title onto a base figure title, e.g. 'model_key: Validation rewards'."""
    if title is None or str(title) == '':
        return base
    return '%s: %s' % (str(title), base)


def plot_validation_rewards(network, title=None, train_step_range=None, ax=None, save_path=None):
    assert len(network.val_reward_history) > 0, 'Network must contain a stored val_loss_history'

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
        fig.suptitle(_prefix_title(title, 'Validation rewards'))
        fig.tight_layout()
        _save_or_show(fig, save_path)
    else:
        ax.plot(train_steps, val_reward_history, label='Validation rewards', color='r', linewidth=1)
        ax.set_xlabel('Training steps')


def plot_final_q_vals(network, environments, title=None, save_path=None):
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
    fig.suptitle(_prefix_title(title, 'Final Q Values'), fontsize=16)
    _save_or_show(fig, save_path)


def plot_actions_over_training(network, environments, title=None, save_path=None):
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
            ax.axvspan(environment_j.reward_position - 0.5, environment_j.reward_position + 0.5,
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

    fig.suptitle(_prefix_title(title, 'Agent licks over training'))
    fig.tight_layout()
    _save_or_show(fig, save_path)


def _treadmill_zone_spans(environment):
    """
    Recover the contiguous cue and reward zones of a Cue_Treadmill from its one-hot state encoding.
    Returns (cue_span, reward_spans) where each span is a (low, high) tuple with an exclusive upper
    bound (i.e. cells low..high-1), suitable for drawing box outlines in heatmap coordinates.
    """
    rep = np.asarray(environment.state_representations)
    total_cues = environment.total_cues

    def span(col):
        idx = np.where(rep[:, col] == 1)[0]
        if len(idx) == 0:
            return None
        return (int(idx.min()), int(idx.max()) + 1)

    cue_span = span(1 + environment.cue_number)
    reward_spans = []
    for i in range(total_cues):
        reward_span = span(1 + total_cues + i)
        if reward_span is not None:
            reward_spans.append(reward_span)
    return cue_span, reward_spans


def plot_hidden_state_cross_correlation(network, environments, population_name=None, title=None,
                                        normalize=False, save_path=None):
    """
    Plot the cross-correlation between hidden-population representations of the trained network across
    pairs of treadmills, following the style of the Spruston lab OSM paper figures
    (fig_4_f_Hebbian_RNN.ipynb). For each pair (i, j), entry [a, b] of the heatmap is the Pearson
    correlation (across hidden units) between the activity vector at position a of treadmill i and
    position b of treadmill j.

    :param network: trained Q_Network
    :param environments: list of Cue_Treadmill environments
    :param population_name: optional str fullname of hidden population (e.g. 'H1E'). Defaults to the
        first population of the layer feeding the output layer.
    :param title: optional str appended to the figure title
    :param normalize: bool; if True (default) each pair's correlation matrix is min-max rescaled across
        all location pairs onto the full [-1, 1] colormap range (weakest match at -1, strongest at +1)
        to maximize contrast/stratification.
    :param save_path: optional str; if provided the figure is saved (high dpi) instead of shown.
    """
    activities = network.get_treadmill_hidden_activity(environments, population_name=population_name)

    n_env = len(environments)
    pairs = [(i, j) for i in range(n_env) for j in range(i + 1, n_env)]
    if len(pairs) == 0:
        print('plot_hidden_state_cross_correlation: need at least two environments; skipping')
        return

    # Cue/reward zones (shared geometry across treadmills) used to annotate every panel
    cue_span, reward_spans = _treadmill_zone_spans(environments[0])
    zones = ([cue_span] if cue_span is not None else []) + reward_spans
    boundary_lines = sorted({bound for span in zones for bound in span} | {environments[0].length})

    icefire = sns.color_palette('icefire', as_cmap=True)

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 4.4), squeeze=False)
    axes = axes[0]

    im = None
    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        t1 = activities[i]  # [length_i, n_units]; rows of the heatmap
        t2 = activities[j]  # [length_j, n_units]; columns of the heatmap
        length_i, length_j = t1.shape[0], t2.shape[0]

        corr = np.zeros((length_i, length_j))
        for a in range(length_i):
            zeroi = np.nan_to_num(t1[a])
            for b in range(length_j):
                onej = np.nan_to_num(t2[b])
                corr[a, b] = stats.pearsonr(zeroi, onej)[0]

        if normalize:
            # min-max rescale onto the full [-1, 1] colormap range to maximize contrast
            min_corr, max_corr = np.nanmin(corr), np.nanmax(corr)
            if max_corr > min_corr:
                corr = 2. * (corr - min_corr) / (max_corr - min_corr) - 1.

        sns.heatmap(corr, ax=ax, cmap=icefire, vmin=-1, vmax=1, cbar=False, rasterized=True)
        im = ax.collections[0]

        # thick black border frame
        ax.axhline(y=0, color='k', linewidth=5)
        ax.axhline(y=corr.shape[0], color='k', linewidth=5)
        ax.axvline(x=0, color='k', linewidth=5)
        ax.axvline(x=corr.shape[1], color='k', linewidth=5)

        # white dashed divider lines at zone boundaries
        for line in boundary_lines:
            ax.axvline(line, linestyle=(0, (2, 5)), color='white', linewidth=1.5)
            ax.axhline(line, linestyle=(0, (2, 5)), color='white', linewidth=1.5)

        # white box outlines around the cue and reward zones
        for (low, high) in zones:
            ax.plot([low, high, high, low, low], [low, low, high, high, low], color='white', linewidth=3)

        ax.set_aspect('equal')
        ax.axis('off')
        ax.grid(False)
        ax.set_title('Treadmill {} (rows) vs {} (cols)'.format(i + 1, j + 1))

    if im is not None:
        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.04, pad=0.04)

    fig.suptitle(_prefix_title(title, 'Cross-correlation of hidden states'))
    _save_or_show(fig, save_path)


def plot_treadmill_hidden_activity(network, environments, population_names=('H1E', 'H2E'), title=None,
                                   save_path=None):
    """
    Plot hidden-population activity as a neuron-by-position heatmap, comparing all treadmills side by
    side. For each population in ``population_names`` a separate figure is produced with one panel per
    treadmill; each panel is a [n_units, length] heatmap (hidden unit on the y-axis, treadmill position
    on the x-axis) drawn with the shared ``icefire`` colormap and a common color scale so activity is
    directly comparable across treadmills. Units are sorted by the position of their peak activity on
    the first treadmill (as in place-cell sequence plots), and that same ordering is applied to every
    panel so a preserved sequence is visible across treadmills. The cue position and each treadmill's
    reward zone are marked with dashed lines.

    :param network: trained Q_Network
    :param environments: list of Cue_Treadmill environments
    :param population_names: iterable of str fullnames of hidden populations to plot (e.g. ('H1E', 'H2E'))
    :param title: optional str appended to each figure title
    :param save_path: optional str; if provided figures are saved (the population name is inserted before
        the file extension, e.g. '..._H1E.png') instead of shown.
    """
    n_env = len(environments)
    # reversed grayscale: zero activity is white, increasing activity darkens toward black, for
    # maximum contrast (black activity on a white background)
    cmap = plt.get_cmap('gray_r')

    for population_name in population_names:
        activities = network.get_treadmill_hidden_activity(environments, population_name=population_name)

        # shared color scale across treadmills; pin vmin to 0 so zero activity is black
        vmin = 0.
        vmax = max(np.nanmax(a) for a in activities)

        # sort units by peak position on the first treadmill (place-cell style ordering), applied to all
        order = np.argsort(np.argmax(np.nan_to_num(activities[0]), axis=0))

        fig, axes = plt.subplots(1, n_env, figsize=(3.2 * n_env, 3.4), squeeze=False, sharey=True)
        axes = axes[0]

        im = None
        for env_idx, (activity, ax) in enumerate(zip(activities, axes)):
            # transpose to [n_units, length] and apply the shared sequence ordering
            im = ax.imshow(activity[:, order].T, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                           interpolation='nearest')
            cue_span, reward_spans = _treadmill_zone_spans(environments[env_idx])
            if cue_span is not None:
                ax.axvline(cue_span[0] - 0.5, color='tab:blue', linestyle='--', linewidth=0.75)
            for (low, high) in reward_spans:
                ax.axvline(low - 0.5, color='tab:red', linestyle='--', linewidth=0.75)
                ax.axvline(high - 0.5, color='tab:red', linestyle='--', linewidth=0.75)
            ax.set_xlabel('Treadmill position')
            ax.set_title('Treadmill %i' % (env_idx + 1))
        axes[0].set_ylabel('Hidden unit (sorted by peak)')

        if im is not None:
            fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04, label='Activity')

        fig.suptitle(_prefix_title(title, 'Hidden unit activity (%s)' % population_name))
        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            _save_or_show(fig, '%s_%s%s' % (base, population_name, ext))
        else:
            _save_or_show(fig, None)


def plot_equilibration_dynamics(network, environments, env_idx=0, position=None, store_num_steps=None, title=None):
    """
    Plot the within-trial equilibration dynamics of each population's activity for a single observation
    drawn from a treadmill environment. This is the RL analog of
    ``EIANN.utils.check_equilibration_dynamics``, which relies on a torch DataLoader to provide a batch
    of inputs. Here the input is instead a single observation taken from one Cue_Treadmill environment.

    The agent is walked from the start of the treadmill up to ``position``: a forward pass is run at
    every position along the way, with ``reinit=True`` only at position 0 so the recurrent state
    accumulated over the preceding positions is carried forward (matching how ``network.test`` traverses
    a treadmill). Step-by-step dynamics are recorded only at the final target position, so the plotted
    settling reflects the recurrent state the network actually arrives at, not an isolated reinitialized
    pass. The average population activity is then plotted across the network's recurrent ``forward_steps``,
    one subplot per population (rows) and per layer (cols, excluding the input layer).

    :param network: trained Q_Network
    :param environments: list of Cue_Treadmill environments
    :param env_idx: int index into ``environments`` selecting which treadmill to probe
    :param position: int treadmill position to record settling at; defaults to the cue position
    :param store_num_steps: int number of trailing forward steps to record; defaults to all forward_steps
    :param title: optional str appended to the figure title
    """
    environment = environments[env_idx]
    if position is None:
        position = environment.cue_position

    # walk from the start up to the target position, carrying recurrent state forward (reinit only at
    # position 0). The agent advances one position per step regardless of action, and actions are not
    # fed back into the network, so stepping through observations by index reproduces the traversal.
    environment.reset()
    for state in range(position + 1):
        current_observation = environment.get_observation(state)
        obs_tensor = torch.tensor(current_observation, dtype=torch.float32).unsqueeze(0)
        # only record dynamics at the final position; reinit (which clears forward_steps_activity)
        # fires solely at position 0, leaving a clean trace for the target forward pass
        store_state = state == position
        network.forward(obs_tensor, store_dynamics=store_state, store_num_steps=store_num_steps,
                        no_grad=True, reinit=(state == 0))

    max_rows = 1
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))
    cols = len(network.layers) - 1
    fig, axes = plt.subplots(max_rows, cols, figsize=(3.2 * cols, 3. * max_rows), squeeze=False)

    for i, layer in enumerate(network):
        if i == 0:
            continue
        col = i - 1
        for row, population in enumerate(layer):
            if population.forward_steps_activity:
                # for memory efficiency, reduce each stored step to its population mean
                average_activity = torch.tensor(
                    [torch.mean(step) for step in population.forward_steps_activity])
                ax = axes[row][col]
                ax.plot(average_activity)
                ax.set_xlabel('Equilibration time steps')
                ax.set_ylabel('Average population activity')
                ax.set_title('%s.%s' % (layer.name, population.name))
                ax.set_ylim((0., ax.get_ylim()[1]))
            population.forward_steps_activity = []
        # hide any unused axes in this column's grid
        for row in range(len(layer.populations), max_rows):
            axes[row][col].axis('off')

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)
    fig.suptitle('Activity equilibration dynamics (treadmill %i, position %i)%s' %
                 (env_idx + 1, position, title_str))
    fig.tight_layout()
    plt.show(block=False)

