# Imports
import torch
import torch.nn.functional as F
import itertools

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


def unpack_data(data_file_path):
    activity_dict = {}
    weight_dict = {}
    metrics_dict = {}
    hyperparams_dict = {}

    with h5py.File(data_file_path, 'r') as f:
        description_list = list(f.keys())

        for description in description_list:
            activity_dict[description] = {}
            weight_dict[description] = {}
            metrics_dict[description] = {}
            hyperparams_dict[description] = {}

            model_group = f[description]
            for key, value in model_group.attrs.items():
                hyperparams_dict[key] = value

            group = model_group['activity_dict']
            for layer in group:
                activity_dict[description][int(layer)] = {}
                for population in group[layer]:
                    if np.all(np.isnan(group[layer][population]) == False):
                        activity_dict[description][int(layer)][population] = group[layer][population][:]


            group = model_group['weight_dict']
            for layer in group:
                weight_dict[description][int(layer)] = {}
                for population in group[layer]:
                    if np.all(np.isnan(group[layer][population]) == False):
                        weight_dict[description][int(layer)][population] = group[layer][population][:]

            group = model_group['metrics_dict']
            for metric in group:
                metrics_dict[description][metric] = group[metric][:]

    return  activity_dict, weight_dict, metrics_dict, hyperparams_dict


def plot_activity(activity_dict, title_dict, model_list, show=False, save=False):

    for model_name in model_list:
        fig = plt.figure(figsize=(10, 7))
        axes = gs.GridSpec(nrows=2, ncols=3,
                           left=0.07, right=0.98,
                           top=0.83, bottom=0.1,
                           wspace=0.3, hspace=0.5)

        for layer in range(len(activity_dict[model_name])):
            for i, pop in enumerate(['E', 'I']):
                if pop not in activity_dict[model_name][layer]:
                    continue
                if np.all(np.isnan(activity_dict[model_name][layer][pop])):
                    continue
                ax = fig.add_subplot(axes[i,layer])
                im = ax.imshow(activity_dict[model_name][layer][pop], aspect='auto', interpolation='none', vmin=0.)
                cbar = plt.colorbar(im, ax=ax)

                if layer == 0:
                    layer_title = 'Input layer'
                elif layer == len(activity_dict[model_name]) - 1:
                    layer_title = 'Output layer'
                else:
                    layer_title = 'Hidden layer'
                ax.set_title('%s (%s)' % (layer_title, pop))
                ax.set_xlabel('Input pattern')
                ax.set_ylabel('Unit ID')
                if activity_dict[model_name][layer][pop].shape[0] == 1:
                    ax.set_xticks([0])
                clean_axes(ax)

        fig.suptitle('%s\n\nActivity' % title_dict[model_name])
        if save:
            fig.savefig(f'figures/activity_{model_name}.svg', edgecolor='white', dpi=300, facecolor='white',
                        transparent=True)

        if show:
            fig.show()


def analyze_activity(activity_dict):
    """
    Sparsity metric from (Vinje & Gallant 2000): https://www.science.org/doi/10.1126/science.287.5456.1273
    :param activity_dict:
    :return: tuple of dict
    """
    sparsity_dict = {}
    discriminability_dict = {}

    for model_name in activity_dict:
        if len(activity_dict[model_name]) > 2:
            num_patterns = activity_dict[model_name][0]['E'].shape[1]
            sparsity = np.zeros(num_patterns)
            for i, pop_activity in enumerate(activity_dict[model_name][1]['E'].T):
                n = pop_activity.shape[0]
                activity_fraction = (np.sum(pop_activity) / n) ** 2 / np.sum(pop_activity ** 2 / n)
                sparsity[i] = (1 - activity_fraction) / (1 - 1 / n)
            sparsity_dict[model_name] = np.nanmean(sparsity)

            # Compute discriminability
            silent_pattern_idx = np.where(np.count_nonzero(activity_dict[model_name][1]['E'], axis=0) == 0)[0]
            similarity_matrix = cosine_similarity(activity_dict[model_name][1]['E'].T)
            similarity_matrix[silent_pattern_idx,: ] = 1
            similarity_matrix[:, silent_pattern_idx] = 1
            similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1) # extract all values below diagonal
            similarity = similarity_matrix[similarity_matrix_idx]
            discriminability = 1 - similarity
            discriminability_dict[model_name] = np.mean(discriminability)

    # Compute ideal sparsity
    ideal_2hot_activity = n_hot_patterns(2, 7)
    sparsity = np.zeros(num_patterns)
    for i, pop_activity in enumerate(ideal_2hot_activity.T):
        n = pop_activity.shape[0]
        activity_fraction = (np.sum(pop_activity) / n) ** 2 / np.sum(pop_activity ** 2 / n)
        sparsity[i] = (1 - activity_fraction) / (1 - 1 / n)
    sparsity_dict['ideal'] = np.nanmean(sparsity)

    # Compute ideal discriminability
    similarity_matrix = cosine_similarity(ideal_2hot_activity.T)
    similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1)  # extract all values below diagonal
    similarity = similarity_matrix[similarity_matrix_idx]
    discriminability = 1 - similarity
    discriminability_dict['ideal'] = np.mean(discriminability)

    return sparsity_dict, discriminability_dict


def plot_activation_funcs(show=False, save=False):
    fig, axis = plt.subplots(figsize=(4.25, 3.75))
    x = torch.linspace(-1.5, 1.5, 1000)
    plt.plot(np.zeros_like(x), x, '--', c='lightgrey')
    plt.plot(x, np.zeros_like(x), '--', c='lightgrey')
    plt.plot(x, F.softplus(x, beta=4), color='k', label='Soft+')
    plt.plot(x, F.relu(x), color='orange', label='ReLU')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-0.25, 1.5])
    plt.ylabel('Output activity')
    plt.xlabel('Summed input')
    plt.title('Activation functions')
    plt.legend(frameon=False)
    fig.tight_layout()
    clean_axes(axis)

    if save:
        fig.savefig(f'figures/activation_func.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)

    if show:
        fig.show()


def plot_metrics(metrics_dict, legend_dict, model_list, show=False, save=False, save_name=None):
    max_index = 0
    for model_name in model_list:
        accuracy = metrics_dict[model_name]['accuracy']
        stable_blocks = 0
        max_accuracy = np.max(accuracy)
        for stable_index, val in enumerate(accuracy):
            if val == max_accuracy:
                stable_blocks += 1
            else:
                stable_blocks = 0
            if stable_blocks == 50:
                break
        max_index = max(stable_index, max_index)

    fig, ax = plt.subplots()
    for model_name in model_list:
        accuracy = metrics_dict[model_name]['accuracy'][:max_index]
        ax.plot(accuracy, label=legend_dict[model_name][0], color=legend_dict[model_name][1])
        ax.set_xlabel('Training blocks')
        ax.set_ylabel('% Correct')
        ax.set_ylim([0, 105])
        # ax.set_xlim(xlim)
        ax.set_title('Argmax Accuracy')
        ax.legend(frameon=False)
    fig.tight_layout()
    clean_axes(ax)

    if save:
        if save_name is None:
            fig.savefig('figures/accuracy_%s.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
        else:
            fig.savefig('figures/accuracy_%s.svg' % save_name, edgecolor='white', dpi=300, facecolor='white',
                        transparent=True)

    if show:
        fig.show()


def n_hot_patterns(n,length):
    all_permutations = np.array(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = np.sum(all_permutations,axis=1)
    idx = np.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns.T


def plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, model_list, show=False, save=False,
                            save_name=None):

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.75))
    ax1 = axes[0]
    ax2 = axes[1]

    xlim = [-0.75, len(model_list) - 0.25]
    ax1.plot(xlim, np.ones_like(xlim) * sparsity_dict['ideal'], '--', color='grey')
    ax2.plot(xlim, np.ones_like(xlim) * discriminability_dict['ideal'], '--', color='grey')

    for x, model_name in enumerate(model_list):
        ax1.bar(x, sparsity_dict[model_name], width=0.5, color=legend_dict[model_name][1])
        ax2.bar(x, discriminability_dict[model_name], width=0.5, color=legend_dict[model_name][1])

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

    if save:
        if save_name is None:
            fig.savefig(f'figures/summary_comparison.svg', edgecolor='white', dpi=300, facecolor='white',
                        transparent=True)
        else:
            fig.savefig(f'figures/summary_comparison_%s.svg' % save_name, edgecolor='white', dpi=300, facecolor='white',
                        transparent=True)

    if show:
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
        ax.set_xlabel('Pre (I)')
        ax.set_ylabel('Post (E)')

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


@click.command()
@click.option("--show", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
def main(show, save, data_dir):

    activity_model_list = ['FBI_RNN_1hidden_tau_Inh7', 'FBI_RNN_1hidden_tau_global_Inh', 'FF_network_1hidden',
                           'FF_network_1hidden_relu', 'Hebb_1_hidden_inh_1_7', 'Hebb_1_hidden_inh_7_7',
                           'Hebb_no_hidden_inh_1', 'Hebb_no_hidden_inh_7', 'btsp_network']

    title_dict = {'FBI_RNN_1hidden_tau_Inh7': 'Backprop - Tuned Inhibition',
                  'FBI_RNN_1hidden_tau_global_Inh': 'Backprop - Global Inhibition',
                  'FF_network_1hidden': 'Standard Backprop',
                  'FF_network_1hidden_relu': 'Standard Backprop - ReLU activation',
                  'Hebb_1_hidden_inh_1_7': 'Hebb - Global Inhibition',
                  'Hebb_1_hidden_inh_7_7': 'Hebb - Tuned Inhibition',
                  'Hebb_no_hidden_inh_1': 'Hebb - Global Inhibition',
                  'Hebb_no_hidden_inh_7': 'Hebb - Tuned Inhibition',
                  'btsp_network': 'Top-Down Dendritic Gating'}

    legend_dict =  {'FBI_RNN_1hidden_tau_Inh7': ('Backprop - Tuned Inh', 'b'),
                    'FBI_RNN_1hidden_tau_global_Inh': ('Backprop - Global Inh', 'grey'),
                    'FF_network_1hidden': ('Standard Backprop', 'k'),
                    'Hebb_1_hidden_inh_1_7': ('Hebb - Global Inh', 'purple'),
                    'Hebb_1_hidden_inh_7_7': ('Hebb - Tuned Inh', 'r'),
                    'btsp_network': ('Dendritic Gating', 'c')}

    ReLU_legend_dict = {'FF_network_1hidden': ('Backprop - Soft+ activation', 'k'),
                        'FF_network_1hidden_relu': ('Backprop - ReLU activation', 'orange')}

    activity_dict_bp, weight_dict_bp, metrics_dict_bp, hyperparams_dict_bp = \
        unpack_data(data_dir+'/'+'20220504_backprop_network_data.hdf5')
    activity_dict_hebb, weight_dict_hebb, metrics_dict_hebb, hyperparams_dict_hebb = \
        unpack_data(data_dir+'/'+'20220405b_Hebb_lateral_inh_network_data.hdf5')
    activity_dict_btsp, weight_dict_btsp, metrics_dict_btsp, hyperparams_dict_btsp = \
        unpack_data(data_dir+'/'+'20220505_150053_btsp_network_data.hdf5')

    activity_dict = {**activity_dict_bp, **activity_dict_hebb, **activity_dict_btsp}
    weight_dict = {**weight_dict_bp, **weight_dict_hebb, **weight_dict_btsp}
    metrics_dict = {**metrics_dict_bp, **metrics_dict_hebb, **metrics_dict_btsp}

    plot_activity(activity_dict, title_dict, activity_model_list, show, save)

    accuracy_model_list_1 = ['FF_network_1hidden','FBI_RNN_1hidden_tau_global_Inh','FBI_RNN_1hidden_tau_Inh7']
    accuracy_model_list_2 = ['FF_network_1hidden', 'FF_network_1hidden_relu']
    accuracy_model_list_3 = ['FF_network_1hidden', 'FBI_RNN_1hidden_tau_Inh7', 'Hebb_1_hidden_inh_7_7', 'btsp_network']

    plot_activation_funcs(show, save)

    plot_metrics(metrics_dict, legend_dict, accuracy_model_list_1, show, save, 1)
    plot_metrics(metrics_dict, ReLU_legend_dict, accuracy_model_list_2, show, save, 2)
    plot_metrics(metrics_dict, legend_dict, accuracy_model_list_3[:-1], show, save, 3)
    plot_metrics(metrics_dict, legend_dict, accuracy_model_list_3, show, save, 4)

    sparsity_dict, discriminability_dict = analyze_activity(activity_dict)

    metrics_model_list_1 = ['FF_network_1hidden','FBI_RNN_1hidden_tau_global_Inh','FBI_RNN_1hidden_tau_Inh7']
    metrics_model_list_2 = ['FF_network_1hidden', 'FBI_RNN_1hidden_tau_Inh7', 'Hebb_1_hidden_inh_7_7', 'btsp_network']
    plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, metrics_model_list_1, show, save, 1)
    plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, metrics_model_list_2[:-1], show, save, 2)
    plot_summary_comparison(sparsity_dict, discriminability_dict, legend_dict, metrics_model_list_2, show, save, 3)

    weights_model_list = ['FBI_RNN_1hidden_tau_Inh7', 'Hebb_1_hidden_inh_7_7']
    plot_lateral_weights(weight_dict, title_dict, weights_model_list, show, save)

    if show:
        plt.show()

    globals().update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)