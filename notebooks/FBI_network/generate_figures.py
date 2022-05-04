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


def plot_metrics(metrics_dict_bp,metrics_dict_hebb,metrics_dict_btsp):



    for model_name in metrics_dict.keys():
        print(model_name)

        accuracy = metrics_dict[model_name]['accuracy']
        loss = metrics_dict[model_name]['loss']
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(loss)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].set_ylim(bottom=0)
        ax[0].set_title('learning curve', fontsize=20)

        ax[1].plot(accuracy)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('% correct')
        ax[1].set_ylim([0, 100])
        ax[1].set_title('Argmax accuracy', fontsize=20)

        plt.suptitle(model_name, fontsize=20)
        m = np.max(accuracy)
        print(f"Epochs to {np.round(m)}% accuracy: {np.where(accuracy >= m)[0][0]}")
        for i in range(accuracy.shape[0]):
            if np.all(accuracy[i:i + 50] == 100):
                print(f"Stable epoch: {i + 50} /n")
                break

        fig.tight_layout()
        sns.despine()


def plot_activity(activity_dict):
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

    legend_dict = {}

    activity_dict_bp, weight_dict_bp, metrics_dict_bp, hyperparams_dict_bp = unpack_data('20220504_backprop_network_data.hdf5')
    activity_dict_hebb, weight_dict_hebb, metrics_dict_hebb, hyperparams_dict_hebb = unpack_data('20220405_Hebb_lateral_inh_network_data.hdf5')
    activity_dict_btsp, weight_dict_btsp, metrics_dict_btsp, hyperparams_dict_btsp = unpack_data('20220504_104630_btsp_network_data.hdf5')

    activity_dict = {**activity_dict_bp, **activity_dict_hebb, **activity_dict_btsp}
    weight_dict = {**weight_dict_bp, **weight_dict_hebb, **weight_dict_btsp}
    metrics_dict = {**metrics_dict_bp, **metrics_dict_hebb, **metrics_dict_btsp}

    globals().update(locals())

    # plot_metrics(metrics_dict)

    if plot:
        plt.show()


if __name__ == '__main__':
    main(standalone_mode=False)