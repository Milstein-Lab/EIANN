# Imports
import numpy as np
import click
import h5py
from copy import deepcopy

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


def plot_activity(activity_dict):
    # for model_name in activity_dict.keys():
    return


def plot_weights(weights_dict):
    '''
    Plot weight matrix for each projection and scatter correlation of recurrent E and I connections
    :return:
    '''
    return


@click.command()
@click.option("--plot", is_flag=True)

def main(plot):

    legend_dict =  {'FBI_RNN_1hidden_tau_Inh7': ('Backprop inh=7','red'),
                    'FBI_RNN_1hidden_tau_Inh7_relu': ('Backprop inh=7, relu','green'),
                    'FBI_RNN_1hidden_tau_global_Inh': ('Backprop global inh','blue'),
                    'FBI_RNN_1hidden_tau_global_Inh_relu': ('Backprop global inh, relu','magenta'),
                    'FF_network_1hidden': ('Backprop FF','cyan'),
                    'FF_network_1hidden_relu': ('Backprop FF, relu','purple'),
                    '_Hebb_1_hidden_inh_1_7': ('Hebb inh=1,7','orange'),
                    '_Hebb_1_hidden_inh_7_7': ('Hebb inh=7,7','maroon'),
                    '_Hebb_no_hidden_inh_1': ('Hebb inh=1, no hidden','tan'),
                    '_Hebb_no_hidden_inh_7': ('Hebb inh=7, no hidden','plum'),
                    'btsp_network': ('BTSP','navy')}

    activity_dict_bp, weight_dict_bp, metrics_dict_bp, hyperparams_dict_bp = unpack_data('20220504_backprop_network_data.hdf5')
    activity_dict_hebb, weight_dict_hebb, metrics_dict_hebb, hyperparams_dict_hebb = unpack_data('20220405_Hebb_lateral_inh_network_data.hdf5')
    activity_dict_btsp, weight_dict_btsp, metrics_dict_btsp, hyperparams_dict_btsp = unpack_data('20220504_104630_btsp_network_data.hdf5')

    activity_dict = {**activity_dict_bp, **activity_dict_hebb, **activity_dict_btsp}
    weight_dict = {**weight_dict_bp, **weight_dict_hebb, **weight_dict_btsp}
    metrics_dict = {**metrics_dict_bp, **metrics_dict_hebb, **metrics_dict_btsp}

    globals().update(locals())

    fig1_list = ['FF_network_1hidden','FBI_RNN_1hidden_tau_global_Inh','FBI_RNN_1hidden_tau_Inh7']
    fig2_list = ['FF_network_1hidden', 'FF_network_1hidden_relu']
    fig3_list = ['FBI_RNN_1hidden_tau_Inh7', '_Hebb_1_hidden_inh_1_7']
    fig4_list = ['FF_network_1hidden', 'FBI_RNN_1hidden_tau_Inh7', '_Hebb_1_hidden_inh_7_7', 'btsp_network']
    fig_superlist = [fig1_list,fig2_list,fig3_list,fig4_list]
    plot_metrics(metrics_dict, legend_dict, fig_superlist, xlim=(0,200))

    if plot:
        plt.show()


if __name__ == '__main__':
    main(standalone_mode=False)