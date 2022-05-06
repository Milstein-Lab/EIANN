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
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
sns.set_style(style='ticks')


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


def plot_activity(activity_dict, legend_dict, plot):
    sparsity_dict = {}
    discriminability_dict = {}
    storage_capacity_dict = {}

    for model_name in activity_dict:
        fig = plt.figure(figsize=(10, 7))
        axes = gs.GridSpec(nrows=2, ncols=3,
                           left=0.05, right=0.98,
                           top=0.83, bottom=0.1,
                           wspace=0.3, hspace=0.5)

        for layer in activity_dict[model_name]:
            for i,pop in enumerate(activity_dict[model_name][layer]):
                ax = fig.add_subplot(axes[i,layer])
                im = ax.imshow(activity_dict[model_name][layer][pop], aspect='auto', cmap='viridis')
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('activity', rotation=270, labelpad=10, fontsize=10)
                cbar.ax.tick_params(labelsize=10)
                ax.set_title(f"layer {layer}, {pop}")
                ax.set_xlabel('Pattern')
                ax.set_ylabel('Unit')

        if "no_hidden" not in model_name:
            # Sparsity metric from (Vinje & Gallant 2000): https://www.science.org/doi/10.1126/science.287.5456.1273
            num_patterns = activity_dict[model_name][0]['E'].shape[1]
            sparsity = np.zeros(num_patterns)
            for i, pop_activity in enumerate(activity_dict[model_name][1]['E'].T):
                n = pop_activity.shape[0]
                activity_fraction = (np.sum(pop_activity) / n) ** 2 / np.sum(pop_activity ** 2 / n)
                sparsity[i] = (1 - activity_fraction) / (1 - 1 / n)
            sparsity_dict[model_name] = np.nanmean(sparsity)

            # Calculate combinatorial patterns with binomial formula (generalized for non-integer values)
            n = activity_dict[model_name][1]['E'].shape[0]
            k = (1 - np.nanmean(sparsity)) * n
            n_choose_k = math.gamma(n+1) / (math.gamma(k+1) * math.gamma(n-k+1))
            storage_capacity_dict[model_name] = n_choose_k / num_patterns

            # Compute discriminability
            activity = activity_dict[model_name][1]['E'].T
            similarity_matrix = cosine_similarity(activity)
            zero_patterns = np.where(np.sum(activity,axis=1)==0)[0]
            similarity_matrix[:,zero_patterns], similarity_matrix[zero_patterns,:] = 1,1
            similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1) # extract all values below diagonal
            similarity = similarity_matrix[similarity_matrix_idx]
            discriminability = 1 - similarity
            discriminability_dict[model_name] = np.mean(discriminability)

            ax = fig.add_subplot(axes[1,0])
            im = ax.imshow(similarity_matrix, cmap='hot')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('cosine similarity', rotation=270, labelpad=10, fontsize=10)
            cbar.ax.tick_params(labelsize=10)
            ax.set_title("Hidden E similarity matrix")

        plt.suptitle(f"Activity \n{legend_dict[model_name][0]}", fontsize=15)
        sns.despine()

        fig.savefig(f'figures/activity_{model_name}.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
        fig.savefig(f'figures/activity_{model_name}.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)

        if plot == False:
            fig.clear()
            plt.close(fig)

    # Compute ideal sparsity
    ideal_2hot_activity = n_hot_patterns(2, 7)
    sparsity = np.zeros(num_patterns)
    for i, pop_activity in enumerate(ideal_2hot_activity.T):
        n = pop_activity.shape[0]
        activity_fraction = (np.sum(pop_activity) / n) ** 2 / np.sum(pop_activity ** 2 / n)
        sparsity[i] = (1 - activity_fraction) / (1 - 1 / n)
    sparsity_dict['ideal'] = np.nanmean(sparsity)

    # Compute ideal combinatorial patterns with binomial formula (generalized for non-integer values)
    n = 7
    k = (1 - np.nanmean(sparsity)) * n
    n_choose_k = math.gamma(n+1) / (math.gamma(k+1) * math.gamma(n-k+1))
    storage_capacity_dict['ideal'] = n_choose_k / num_patterns

    # Compute ideal discriminability
    activity = ideal_2hot_activity.T
    similarity_matrix = cosine_similarity(activity)
    zero_patterns = np.where(np.sum(activity, axis=1) == 0)[0]
    similarity_matrix[:, zero_patterns], similarity_matrix[zero_patterns, :] = 1, 1
    similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1)  # extract all values below diagonal
    similarity = similarity_matrix[similarity_matrix_idx]
    discriminability = 1 - similarity
    discriminability_dict['ideal'] = np.mean(discriminability)

    return sparsity_dict, discriminability_dict, storage_capacity_dict


def plot_activation_func():
    fig = plt.figure(figsize=(5,5))
    x = torch.linspace(-2, 1.5, 1000)
    plt.plot(x, F.softplus(x, beta=4), color='blue', label='Softplus')
    plt.plot(x, F.relu(x), color='red', label='ReLU')

    plt.ylim(bottom=-0.5)
    plt.title('ReLU vs Softplus activation')
    plt.legend()
    sns.despine()

    fig.savefig(f'figures/activation_func.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    fig.savefig(f'figures/activation_func.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_metrics(metrics_dict, legend_dict, fig_superlist, xlim=(0,200)):
    fig, ax = plt.subplots(2,2, figsize=(10, 5))

    for row,fig_list in enumerate(fig_superlist):
        col = 0
        if row>1:
            col = 1
            row -= 2
        for model_name in fig_list:
            accuracy = metrics_dict[model_name]['accuracy']
            ax[row,col].plot(accuracy, label=legend_dict[model_name][0], color=legend_dict[model_name][1])
        ax[row,col].set_xlabel('Training blocks')
        ax[row,col].set_ylabel('% correct')
        ax[row,col].set_ylim(bottom=0)
        ax[row,col].set_xlim(xlim)
        ax[row,col].set_title('Argmax accuracy', fontsize=12)
        ax[row,col].legend()

    fig.tight_layout()
    sns.despine()

    fig.savefig('figures/accuracy.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    fig.savefig('figures/accuracy.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def n_hot_patterns(n,length):
    all_permutations = np.array(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = np.sum(all_permutations,axis=1)
    idx = np.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns.T


def plot_summary_comparison(sparsity_dict, discriminability_dict, storage_capacity_dict, legend_dict, fig_superlist):

    fig = plt.figure(figsize=(10, 7))
    axes = gs.GridSpec(nrows=3, ncols=4,
                       left=0.07, right=0.98,
                       top=0.95, bottom=0.15,
                       wspace=0.3, hspace=0.2)

    for i,list in enumerate(fig_superlist):
        ax1 = fig.add_subplot(axes[0,i])
        ax2 = fig.add_subplot(axes[1,i])
        ax3 = fig.add_subplot(axes[2,i])

        for x,model_name in enumerate(list):
            ax1.bar(x, sparsity_dict[model_name], width=0.5, color=legend_dict[model_name][1])
            ax1.set_xlim([-1,4])
            ax1.set_ylim([0,1])
            ax2.bar(x, discriminability_dict[model_name], width=0.5, color=legend_dict[model_name][1])
            ax2.set_xlim([-1,4])
            ax2.set_ylim([0,1])
            ax3.bar(x, storage_capacity_dict[model_name], width=0.5, color=legend_dict[model_name][1])
            ax3.set_xlim([-1,4])

        ax1.plot([-1,4],sparsity_dict['ideal']*np.ones(2),'--', color='gray')
        ax2.plot([-1,4],discriminability_dict['ideal']*np.ones(2),'--', color='gray')
        ax3.plot([-1,4],storage_capacity_dict['ideal']*np.ones(2),'--', color='gray')

        ax1.set_ylabel('Sparsity')
        ax2.set_ylabel('Discriminability')
        ax3.set_ylabel(r'Max storage capacity $\binom{n}{k}$')

        ax1.set_xticks([])
        ax2.set_xticks([])
        ax3.set_xticks(np.arange(len(list)))
        label_list = [legend_dict[x][0] for x in list]
        ax3.set_xticklabels(label_list, rotation=-45, ha="left", rotation_mode="anchor")

    sns.despine()

    fig.savefig(f'figures/summary_comparison.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    fig.savefig(f'figures/summary_comparison.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_weights(weights_dict):
    model_list = ['FBI_RNN_1hidden_tau_Inh7','FBI_RNN_1hidden_tau_Inh7_relu',
                  'Hebb_1_hidden_inh_1_7','Hebb_1_hidden_inh_7_7',
                  'Hebb_no_hidden_inh_1','Hebb_no_hidden_inh_7']
    for model_name in model_list:
        fig = plt.figure(figsize=(12, 3))
        axes = gs.GridSpec(nrows=1, ncols=5, width_ratios=[1,2,1,1,2],
                           left=0.04, right=0.98,
                           top=0.7, bottom=0.2,
                           wspace=1.5, hspace=0.2)

        layer = list(weight_dict[model_name].keys())[-1]

        if "Hebb" in model_name:
            ax = fig.add_subplot(axes[0, 3])
            projection = 'I_I'
            im = ax.imshow(weight_dict[model_name][layer][projection], aspect='equal', cmap='Blues_r')
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.006, ax.get_position().height])
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('weight', rotation=270, labelpad=10, fontsize=10)
            cbar.ax.tick_params(labelsize=10)
            ax.set_title(f"layer {1}, {projection}")
            ax.set_xlabel('Pre')
            ax.set_ylabel('Post')

        ax = fig.add_subplot(axes[0,0])
        projection = 'E_FF'
        im = ax.imshow(weight_dict[model_name][layer][projection], aspect='equal', cmap='Reds')
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.006, ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('weight', rotation=270, labelpad=10, fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(f"layer {1}, {projection}")
        ax.set_xlabel('Pre')
        ax.set_ylabel('Post')

        ax = fig.add_subplot(axes[0,1])
        projection = 'I_E'
        im = ax.imshow(weight_dict[model_name][layer][projection], aspect='equal', cmap='Reds')
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.006, ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('weight', rotation=270, labelpad=10, fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(f"layer {1}, {projection}")
        ax.set_xlabel('Pre')
        ax.set_ylabel('Post')

        ax = fig.add_subplot(axes[0,2])
        projection = 'E_I'
        im = ax.imshow(weight_dict[model_name][layer][projection], aspect='equal', cmap='Blues_r')
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.006, ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('weight', rotation=270, labelpad=10, fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(f"layer {1}, {projection}")
        ax.set_xlabel('Pre')
        ax.set_ylabel('Post')

        # Plot weight correlations
        ax = fig.add_subplot(axes[0,4])
        x = weight_dict[model_name][layer]['I_E'].T.flatten()
        y = weight_dict[model_name][layer]['E_I'].flatten()
        ax.scatter(x, y)
        ax.set_xlabel('E->I weights')
        ax.set_ylabel('I->E weights')

        m, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(np.min(x), np.max(x), 2)
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, c='red')
        r_val, p_val = stats.pearsonr(x, y)
        ax.text(0.5,0,f'r^2={np.round(r_val**2,decimals=4)}')

        fig.suptitle(f'Weights \n {model_name}')
        sns.despine()

        fig.savefig(f'figures/weights_{model_name}.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
        fig.savefig(f'figures/weights_{model_name}.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)



@click.command()
@click.option("--plot", is_flag=True)

def main(plot):

    legend_dict =  {'FBI_RNN_1hidden_tau_Inh7': ('Backprop inh=7','red'),
                    'FBI_RNN_1hidden_tau_Inh7_relu': ('Backprop inh=7, relu','green'),
                    'FBI_RNN_1hidden_tau_global_Inh': ('Backprop global inh','blue'),
                    'FBI_RNN_1hidden_tau_global_Inh_relu': ('Backprop global inh, relu','magenta'),
                    'FF_network_1hidden': ('Backprop FF','cyan'),
                    'FF_network_1hidden_relu': ('Backprop FF, relu','purple'),
                    'Hebb_1_hidden_inh_1_7': ('Hebb inh=1,7','orange'),
                    'Hebb_1_hidden_inh_7_7': ('Hebb inh=7,7','maroon'),
                    'Hebb_no_hidden_inh_1': ('Hebb inh=1, no hidden','tan'),
                    'Hebb_no_hidden_inh_7': ('Hebb inh=7, no hidden','plum'),
                    'btsp_network': ('BTSP','navy')}

    activity_dict_bp, weight_dict_bp, metrics_dict_bp, hyperparams_dict_bp = unpack_data('20220504_backprop_network_data.hdf5')
    activity_dict_hebb, weight_dict_hebb, metrics_dict_hebb, hyperparams_dict_hebb = unpack_data('20220405_Hebb_lateral_inh_network_data.hdf5')
    activity_dict_btsp, weight_dict_btsp, metrics_dict_btsp, hyperparams_dict_btsp = unpack_data('20220504_104630_btsp_network_data.hdf5')

    activity_dict = {**activity_dict_bp, **activity_dict_hebb, **activity_dict_btsp}
    weight_dict = {**weight_dict_bp, **weight_dict_hebb, **weight_dict_btsp}
    metrics_dict = {**metrics_dict_bp, **metrics_dict_hebb, **metrics_dict_btsp}

    # fig1_list = ['FF_network_1hidden','FBI_RNN_1hidden_tau_global_Inh','FBI_RNN_1hidden_tau_Inh7']
    # fig2_list = ['FF_network_1hidden', 'FF_network_1hidden_relu']
    # fig3_list = ['FBI_RNN_1hidden_tau_Inh7', 'Hebb_1_hidden_inh_1_7']
    # fig4_list = ['FF_network_1hidden', 'FBI_RNN_1hidden_tau_Inh7', 'Hebb_1_hidden_inh_7_7', 'btsp_network']
    # fig_superlist = [fig1_list,fig2_list,fig3_list,fig4_list]
    # plot_metrics(metrics_dict, legend_dict, fig_superlist, xlim=(0,200))

    sparsity_dict, discriminability_dict, storage_capacity_dict = plot_activity(activity_dict, legend_dict, plot=False)
    globals().update(locals())

    # plot_weights(weight_dict)

    fig1_list = ['FF_network_1hidden','FBI_RNN_1hidden_tau_global_Inh','FBI_RNN_1hidden_tau_Inh7']
    fig2_list = ['FF_network_1hidden', 'FF_network_1hidden_relu']
    fig3_list = ['FBI_RNN_1hidden_tau_Inh7', 'Hebb_1_hidden_inh_1_7']
    fig4_list = ['FF_network_1hidden', 'FBI_RNN_1hidden_tau_Inh7', 'Hebb_1_hidden_inh_7_7', 'btsp_network']
    fig_superlist = [fig1_list,fig2_list,fig3_list,fig4_list]
    plot_summary_comparison(sparsity_dict, discriminability_dict, storage_capacity_dict, legend_dict, fig_superlist)

    # plot_activation_func()

    if plot:
        plt.show()


if __name__ == '__main__':
    main(standalone_mode=False)