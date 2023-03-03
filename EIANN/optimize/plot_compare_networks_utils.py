# Imports
import torch
import torch.nn.functional as F
import itertools

from EIANN.utils import n_hot_patterns

import numpy as np
import click
import h5py
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.sans-serif'] = 'Arial'


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


def unpack_data(model_list, data_file_path_dict):
    activity_dict = {}
    metrics_dict = {}

    for description in model_list:
        data_file_path = data_file_path_dict[description]
        with h5py.File(data_file_path, 'r') as f:
            activity_dict[description] = {}
            metrics_dict[description] = {'loss': [], 'accuracy': []}
            first_seed_group = next(iter(f[description].values()))

            for post_layer in first_seed_group['activity']:
                activity_dict[description][post_layer] = {}
                for post_pop in first_seed_group['activity'][post_layer]:
                    activity_dict[description][post_layer][post_pop] = []
                    for seed_group in f[description].values():
                        activity_dict[description][post_layer][post_pop].append(
                            seed_group['activity'][post_layer][post_pop][:])

            for seed_group in f[description].values():
                for metric in seed_group['metrics']:
                    metrics_dict[description][metric].append(seed_group['metrics'][metric][:])

    return  activity_dict, metrics_dict


def unpack_data_CL(model_list, data_file_path_dict):
    activity_dict = {}
    metrics_dict = {}

    for description in model_list:
        data_file_path = data_file_path_dict[description]
        with h5py.File(data_file_path, 'r') as f:
            activity_dict[description] = {'phase1': {}, 'phase2': {}}
            metrics_dict[description] = {'phase1_loss': [], 'phase1_accuracy': [],
                                         'phase2_loss': [], 'phase2_accuracy': [],
                                         'final_loss': [], 'final_accuracy': []}
            first_seed_group = next(iter(f[description].values()))
            for post_layer in first_seed_group['phase1_activity']:
                activity_dict[description]['phase1'][post_layer] = {}
                activity_dict[description]['phase2'][post_layer] = {}
                for post_pop in first_seed_group['phase1_activity'][post_layer]:
                    activity_dict[description]['phase1'][post_layer][post_pop] = []
                    activity_dict[description]['phase2'][post_layer][post_pop] = []
                    for seed_group in f[description].values():
                        activity_dict[description]['phase1'][post_layer][post_pop].append(
                            seed_group['phase1_activity'][post_layer][post_pop][:])
                        activity_dict[description]['phase2'][post_layer][post_pop].append(
                            seed_group['phase2_activity'][post_layer][post_pop][:])
            for seed_group in f[description].values():
                metrics_dict[description]['phase1_loss'].append(seed_group['metrics']['phase1_loss'][:])
                metrics_dict[description]['phase2_loss'].append(seed_group['metrics']['phase2_loss'][:])
                metrics_dict[description]['final_loss'].append(seed_group['metrics']['final_loss'][0])
                metrics_dict[description]['phase1_accuracy'].append(seed_group['metrics']['phase1_accuracy'][:])
                metrics_dict[description]['phase2_accuracy'].append(seed_group['metrics']['phase2_accuracy'][:])
                metrics_dict[description]['final_accuracy'].append(seed_group['metrics']['final_accuracy'][0])

    return  activity_dict, metrics_dict


def plot_n_choose_k_task(n=7, k=2):

    layer_label_dict = {'Input': 'Input layer', 'H1': 'Hidden layer', 'Output': 'Target Output'}

    fig = plt.figure(figsize=(10, 7))
    axes = gs.GridSpec(nrows=2, ncols=3,
                       left=0.07, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.5)

    activity_dict = {}
    hidden_activity = n_hot_patterns(n=k, length=n)
    activity_dict['H1'] = torch.tensor(hidden_activity)
    input_size = hidden_activity.shape[1]
    activity_dict['Input'] = torch.eye(input_size)
    activity_dict['Output'] = torch.eye(input_size)

    for li, layer in enumerate(['Input', 'H1', 'Output']):
        ax = fig.add_subplot(axes[0, li])
        im = ax.imshow(activity_dict[layer], aspect='equal',
                       interpolation='none', vmin=0.)
        cbar = plt.colorbar(im, ax=ax)
        ax.set_title('%s' % (layer_label_dict[layer]))
        ax.set_xlabel('Input pattern')
        ax.set_ylabel('Unit ID')
        # clean_axes(ax)
    fig.show()


def plot_initial_activity(activity_dict):

    layer_label_dict = {'Input': 'Input layer', 'H1': 'Hidden layer', 'Output': 'Output layer'}

    fig = plt.figure(figsize=(10, 7))
    axes = gs.GridSpec(nrows=2, ncols=3,
                       left=0.07, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.5)

    for li, layer in enumerate(['Input', 'H1', 'Output']):
        ax = fig.add_subplot(axes[0, li])
        im = ax.imshow(activity_dict[layer], aspect='equal',
                       interpolation='none', vmin=0.)
        cbar = plt.colorbar(im, ax=ax)
        ax.set_title('%s' % (layer_label_dict[layer]))
        ax.set_xlabel('Input pattern')
        ax.set_ylabel('Unit ID')
        # clean_axes(ax)
    fig.show()


def plot_activity(activity_dict, title_dict, example_index_dict, model_list, label_pop=True):
    """

    :param activity_dict:
    :param title_dict:
    :param example_index_dict:
    :param model_list:
    :param label_pop: bool
    """

    pop_label_dict = {'E': 'Exc', 'FBI': 'Inh'}
    layer_label_dict = {'Input': 'Input layer', 'H1': 'Hidden layer', 'Output': 'Output layer'}

    for model_name in model_list:
        fig = plt.figure(figsize=(10, 7))
        axes = gs.GridSpec(nrows=2, ncols=3,
                           left=0.07, right=0.98,
                           top=0.83, bottom=0.1,
                           wspace=0.3, hspace=0.5)

        example_index = example_index_dict[model_name]

        for li, layer in enumerate(['Input', 'H1', 'Output']):
            for i, pop in enumerate(['E', 'FBI']):
                if pop not in activity_dict[model_name][layer]:
                    continue
                ax = fig.add_subplot(axes[i,li])
                im = ax.imshow(activity_dict[model_name][layer][pop][example_index], aspect='equal',
                               interpolation='none', vmin=0.)
                cbar = plt.colorbar(im, ax=ax)
                if label_pop:
                    ax.set_title('%s (%s)' % (layer_label_dict[layer], pop_label_dict[pop]))
                else:
                    ax.set_title('%s' % (layer_label_dict[layer]))
                ax.set_xlabel('Input pattern')
                ax.set_ylabel('Unit ID')
                if activity_dict[model_name][layer][pop][example_index].shape[0] == 1:
                    ax.set_yticks([0])
                # clean_axes(ax)
        fig.suptitle(title_dict[model_name])
        fig.show()

        fig.savefig(f'figures/{model_name}_activity.svg',dpi=300)
        fig.savefig(f'figures/{model_name}_activity.png',dpi=300)


def plot_activity_CL(activity_dict, title_dict, example_index_dict, model_list, label_pop=True):
    """

    :param activity_dict:
    :param title_dict:
    :param example_index_dict:
    :param model_list:
    :param label_pop: bool
    """

    pop_label_dict = {'E': 'Exc', 'FBI': 'Inh'}
    layer_label_dict = {'Input': 'Input layer', 'H1': 'Hidden layer', 'Output': 'Output layer'}

    for model_name in model_list:
        for phase in range(1, 3):
            phase_key = 'phase%i' % phase
            fig = plt.figure(figsize=(10, 7))
            axes = gs.GridSpec(nrows=2, ncols=3,
                               left=0.07, right=0.98,
                               top=0.83, bottom=0.1,
                               wspace=0.3, hspace=0.5)

            example_index = example_index_dict[model_name]

            for li, layer in enumerate(['Input', 'H1', 'Output']):
                for i, pop in enumerate(['E', 'FBI']):
                    if pop not in activity_dict[model_name][phase_key][layer]:
                        continue
                    ax = fig.add_subplot(axes[i,li])
                    im = ax.imshow(activity_dict[model_name][phase_key][layer][pop][example_index], aspect='equal',
                                   interpolation='none', vmin=0.)
                    cbar = plt.colorbar(im, ax=ax)

                    if label_pop:
                        ax.set_title('%s (%s)' % (layer_label_dict[layer], pop_label_dict[pop]))
                    else:
                        ax.set_title('%s' % (layer_label_dict[layer]))
                    ax.set_xlabel('Input pattern')
                    ax.set_ylabel('Unit ID')
                    if activity_dict[model_name][phase_key][layer][pop][example_index].shape[0] == 1:
                        ax.set_yticks([0])
                    # clean_axes(ax)
            fig.suptitle('%s\n\nAfter phase %i' % (title_dict[model_name], phase))
            fig.show()

            fig.savefig(f'figures/{model_name}_{phase}_activity.svg', dpi=300)
            fig.savefig(f'figures/{model_name}_{phase}_activity.png', dpi=300)


def plot_input_patterns_CL(activity_dict, split=0.75):
    """

    :param activity_dict:
    :param split: float
    """

    pop_label_dict = {'E': 'Exc', 'FBI': 'Inh'}
    layer_label_dict = {'Input': 'Input layer', 'H1': 'Hidden layer', 'Output': 'Output layer'}

    model_name = next(iter(activity_dict.keys()))
    phase_key = next(iter(activity_dict[model_name].keys()))
    layer = 'Input'
    pop = 'E'
    example_index = 0
    li = 0

    input_patterns = activity_dict[model_name][phase_key][layer][pop][example_index]
    num_units = input_patterns.shape[0]
    num_samples = input_patterns.shape[1]
    phase_1_num_samples = round(num_samples * split)
    phase_indexes = {1: (0, phase_1_num_samples),
                     2: (phase_1_num_samples, num_samples)}

    fig = plt.figure(figsize=(10, 7))
    axes = gs.GridSpec(nrows=2, ncols=3,
                       left=0.07, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.5)
    for i, phase in enumerate(range(1, 3)):
        ax = fig.add_subplot(axes[li ,i])
        im = ax.imshow(input_patterns[:, phase_indexes[phase][0]:phase_indexes[phase][1]], aspect='equal',
                       interpolation='none', vmin=0.)
        ax.set_xticks([0, phase_indexes[phase][1] - phase_indexes[phase][0] - 1])
        ax.set_xticklabels(['%i' % phase_indexes[phase][0], '%i' % (phase_indexes[phase][1] - 1)])

        ax.set_title('Phase %i\n\n%s' % (phase, layer_label_dict[layer]))
        ax.set_xlabel('Input pattern')
        ax.set_ylabel('Unit ID')
    fig.show()


def analyze_hidden_representations(activity_dict, layer='H1', pop='E'):
    """
    Sparsity metric from (Vinje & Gallant 2000): https://www.science.org/doi/10.1126/science.287.5456.1273
    :param activity_dict:
    :param layer: str
    :param pop: str
    :return: tuple of dict of list of float
    """
    sparsity_dict = {}
    discriminability_dict = {}

    for model_name in activity_dict:
        sparsity_dict[model_name] = []
        discriminability_dict[model_name] = []

        num_patterns = activity_dict[model_name][layer][pop][0].shape[1]
        num_units = activity_dict[model_name][layer][pop][0].shape[0]
        for pop_activity in activity_dict[model_name][layer][pop]:
            activity_fraction = (np.sum(pop_activity, axis=0) / num_units) ** 2 / \
                                np.sum(pop_activity ** 2 / num_units, axis=0)
            sparsity = (1 - activity_fraction) / (1 - 1 / num_units)
            sparsity[np.where(np.sum(pop_activity, axis=0) == 0.)] = 0.
            sparsity_dict[model_name].append(np.nanmean(sparsity))

            # Compute discriminability
            silent_pattern_idx = np.where(np.count_nonzero(pop_activity, axis=0) == 0)[0]
            similarity_matrix = cosine_similarity(pop_activity.T)
            similarity_matrix[silent_pattern_idx,: ] = 1
            similarity_matrix[:, silent_pattern_idx] = 1
            similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1) # extract all values below diagonal
            similarity = similarity_matrix[similarity_matrix_idx]
            discriminability = 1 - similarity
            discriminability_dict[model_name].append(np.mean(discriminability))

    # Compute ideal sparsity
    ideal_2hot_activity = n_hot_patterns(2, 7)
    pop_activity = ideal_2hot_activity
    activity_fraction = (np.sum(pop_activity, axis=0) / num_units) ** 2 / \
                        np.sum(pop_activity ** 2 / num_units, axis=0)
    sparsity = (1 - activity_fraction) / (1 - 1 / num_units)
    sparsity_dict['ideal'] = np.nanmean(sparsity)

    # Compute ideal discriminability
    similarity_matrix = cosine_similarity(ideal_2hot_activity.T)
    similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1)  # extract all values below diagonal
    similarity = similarity_matrix[similarity_matrix_idx]
    discriminability = 1 - similarity
    discriminability_dict['ideal'] = np.mean(discriminability)

    return sparsity_dict, discriminability_dict


def plot_activation_funcs():

    fig, axis = plt.subplots(figsize=(4.25, 3.75))
    x = torch.linspace(-0.5, 0.5, 100)
    plt.plot(np.zeros_like(x), x, '--', c='lightgrey')
    plt.plot(x, np.zeros_like(x), '--', c='lightgrey')
    plt.plot(x, F.softplus(x, beta=10.), color='k', label='Softplus')
    plt.plot(x, F.relu(x), color='orange', label='Rectified')
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.1, 0.5])
    plt.ylabel('Output activity')
    plt.xlabel('Summed input')
    plt.title('Activation functions')
    plt.legend(frameon=False)
    fig.tight_layout()
    clean_axes(axis)

    fig.show()


def plot_metrics(metrics_dict, legend_dict, model_list):

    orig_font_size = mpl.rcParams['font.size']
    # mpl.rcParams['font.size'] = 14.
    fig, ax = plt.subplots(figsize=(5., 4.))
    for model_name in model_list:
        mean_accuracy = np.mean(metrics_dict[model_name]['accuracy'], axis=0)
        std_accuracy = np.std(metrics_dict[model_name]['accuracy'], axis=0)
        epochs = np.arange(0, len(mean_accuracy))
        ax.plot(epochs, mean_accuracy, label=legend_dict[model_name][0], color=legend_dict[model_name][1])
        ax.fill_between(epochs, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy,
                        color=legend_dict[model_name][1], alpha=0.25)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('% Correct')
        ax.set_ylim([0, 110])
        ax.set_title('Accuracy')
        ax.legend(loc='best', frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    fig.show()
    fig.savefig('figures/accuracy.svg', dpi=300)
    fig.savefig('figures/accuracy.png', dpi=300)
    mpl.rcParams['font.size'] = orig_font_size


def plot_metrics_CL(metrics_dict, legend_dict, model_list):

    for phase in range(1, 3):
        fig, ax = plt.subplots()
        for model_name in model_list:
            key = 'phase%i_accuracy' % phase
            mean_accuracy = np.mean(metrics_dict[model_name][key], axis=0)
            std_accuracy = np.std(metrics_dict[model_name][key], axis=0)
            epochs = np.arange(0, len(mean_accuracy))
            ax.plot(epochs, mean_accuracy, label=legend_dict[model_name][0], color=legend_dict[model_name][1])
            ax.fill_between(epochs, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy,
                            color=legend_dict[model_name][1], alpha=0.25)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('% Correct')
            ax.set_ylim([0, 110])
            ax.set_title('Accuracy during phase %i' % phase)
            ax.legend(loc='best', frameon=False)
        clean_axes(ax)
        fig.tight_layout()
        fig.show()
        fig.savefig(f'figures/accuracy_{phase}.svg',dpi=300)
        fig.savefig(f'figures/accuracy_{phase}.png',dpi=300)

    orig_font_size = mpl.rcParams['font.size']
    # mpl.rcParams['font.size'] = 14.
    fig, ax = plt.subplots(2,1,figsize=(4,5))
    xlim = [-0.75, len(model_list) - 0.25]
    key = 'final_accuracy'
    for x, model_name in enumerate(model_list):
        mean_accuracy = np.mean(metrics_dict[model_name][key])
        std_accuracy = np.std(metrics_dict[model_name][key])
        ax[1].bar(x, mean_accuracy, yerr=std_accuracy, width=0.5,
                color=legend_dict[model_name][1], alpha=0.7)
    ax[1].set_xlim(xlim)
    ax[1].set_ylim([0, 100])
    ax[1].set_ylabel('% Correct')
    ax[1].set_title('Final accuracy')
    ax[1].set_xticks(np.arange(len(model_list)))
    label_list = [legend_dict[model_name][0] for model_name in model_list]
    ax[1].set_xticklabels(label_list, rotation=-45, ha="left", rotation_mode="anchor")
    clean_axes(ax[1])

    xlim = [-0.75, len(model_list) - 0.25]
    key = 'phase1_accuracy'
    for x, model_name in enumerate(model_list):
        final_accuracy_phase1 = np.array(metrics_dict[model_name][key])[:,-1]
        mean_accuracy = np.mean(final_accuracy_phase1)
        std_accuracy = np.std(final_accuracy_phase1)
        ax[0].bar(x, mean_accuracy, yerr=std_accuracy, width=0.5,
                color=legend_dict[model_name][1], alpha=0.7)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim([0, 100])
    ax[0].set_ylabel('% Correct')
    ax[0].set_title('Phase 1 accuracy')
    ax[0].set_xticks(np.arange(len(model_list)))
    label_list = [legend_dict[model_name][0] for model_name in model_list]
    ax[0].set_xticklabels(label_list, rotation=-45, ha="left", rotation_mode="anchor")
    clean_axes(ax[0])
    fig.tight_layout()
    fig.show()
    fig.savefig(f'figures/accuracy_bar_CL.svg', dpi=300)
    fig.savefig(f'figures/accuracy_bar_CL.png', dpi=300)
    mpl.rcParams['font.size'] = orig_font_size


def n_hot_patterns(n,length):
    all_permutations = np.array(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = np.sum(all_permutations,axis=1)
    idx = np.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns.T


def plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list):

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.75))
    ax1 = axes[0]
    ax2 = axes[1]

    xlim = [-0.75, len(model_list) - 0.25]
    ax1.plot(xlim, np.ones_like(xlim) * sparsity_dict['ideal'], '--', color='grey')
    ax2.plot(xlim, np.ones_like(xlim) * discriminability_dict['ideal'], '--', color='grey')

    for x, model_name in enumerate(model_list):
        ax1.bar(x, np.mean(sparsity_dict[model_name]), yerr=np.std(sparsity_dict[model_name]), width=0.5,
                color=legend_dict[model_name][1])
        ax2.bar(x, np.mean(discriminability_dict[model_name]), yerr=np.std(discriminability_dict[model_name]),
                width=0.5, color=legend_dict[model_name][1])

    ax1.set_xlim(xlim)
    ax1.set_ylim([0, 1])
    ax2.set_xlim(xlim)
    ax2.set_ylim([0, 1])

    ax1.set_title('Sparsity')
    ax2.set_title('Discriminability')
    ax1.set_xticks(np.arange(len(model_list)))
    ax2.set_xticks(np.arange(len(model_list)))
    label_list = [legend_dict[model_name][0] for model_name in model_list]
    ax1.set_xticklabels(label_list, rotation=-45, ha="left", rotation_mode="anchor")
    ax2.set_xticklabels(label_list, rotation=-45, ha="left", rotation_mode="anchor")
    clean_axes(axes)
    fig.tight_layout()
    fig.show()


def plot_lateral_weights(weight_dict, title_dict, model_list, show=False, save=False):

    for model_name in model_list:
        fig, axes = plt.subplots(1, 3, figsize=(10., 4.))

        layer = sorted(list(weight_dict[model_name].keys()))[-1]

        ax = axes[0]
        projection = 'I_E'
        im = ax.imshow(weight_dict[model_name][layer][projection], aspect='auto', interpolation='none')
        cbar = plt.colorbar(im, ax=ax)
        ax.set_title('I <- E')
        ax.set_xlabel('Pre (E)')
        ax.set_ylabel('Post (I)')

        ax = axes[1]
        projection = 'E_I'
        im = ax.imshow(weight_dict[model_name][layer][projection].T, aspect='auto', interpolation='none')
        cbar = plt.colorbar(im, ax=ax)
        ax.set_title('E <- I')
        ax.set_xlabel('Post (E)')
        ax.set_ylabel('Pre (I)')

        # Plot weight correlations
        ax = axes[2]
        x = weight_dict[model_name][layer]['I_E'].T.flatten()
        y = weight_dict[model_name][layer]['E_I'].flatten()
        ax.scatter(x, y, c='grey')
        ax.set_xlabel('I <- E')
        ax.set_ylabel('E <- I')

        m, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(np.min(x), np.max(x), 2)
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, c='k')
        r_val, p_val = stats.pearsonr(x, y)
        ax.text(0.5,0,f'R^2={np.round(r_val**2,decimals=4)}')

        clean_axes(axes)
        fig.suptitle('%s\n\nOutput layer weights' % title_dict[model_name])
        fig.tight_layout()

        if save:
            fig.savefig(f'figures/weights_{model_name}.svg', edgecolor='white', dpi=300, facecolor='white',
                        transparent=True)

        if show:
            fig.show()
