import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import h5py

import EIANN.utils as ut
import EIANN.plot as pt


def generate_Figure1(all_dataloaders, show=True, save=False):
    '''
    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    All networks have 1 hidden layer with 500 E units and 50 somaI units.
    '''
    plt.rcParams.update({'font.size': 8,
                        'axes.spines.right': False,
                        'axes.spines.top':   False,
                        'axes.linewidth':    0.5,
                        'axes.labelpad':     2.0, 
                        'xtick.major.size':  2,
                        'xtick.major.width': 0.5,
                        'ytick.major.size':  2,
                        'ytick.major.width': 0.5,
                        'legend.frameon':       False,
                        'legend.handletextpad': 0.1,
                        'figure.figsize': [10.0, 3.0],
                        'svg.fonttype': 'none',
                        'text.usetex': False})

    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 180 * mm))
    axes = gs.GridSpec(nrows=4, ncols=4,
                       left=0.06,right=0.98,
                       top=0.96, bottom = 0.08,
                       wspace=0.4, hspace=0.2)
        
    model_dict = {"vanBP":      {"config": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl",
                                 "color": "black",
                                 "name": "Vanilla BP"},
                  "unsup_Hebb": {"config": "20231025_EIANN_1_hidden_mnist_Gjorgjieva_Hebb_config_F_optimized.yaml", 
                                 "pickle": "20230712_EIANN_1_hidden_mnist_Gjorgjieva_Hebb_config_F_66049_257.pkl",
                                 "color": "blue",
                                 "name": "Unsupervised Hebb"},
                  "bpDale":     {"config": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_66049_257.pkl",
                                 "color": "green",
                                 "name": "Backprop with Dale's Law"},
                  "sup_Hebb":   {"config": "20231025_EIANN_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_config_F_optimized.yaml", 
                                 "pickle": "20230505_EIANN_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_config_F_66049_257.pkl",
                                 "color": "red",
                                 "name": "Supervised Hebb"},
                 }


    config_path_prefix = "network_config/mnist/"
    saved_network_path_prefix = "data/mnist/"
    network_seed = 66049
    overwrite = True

    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    for i, model_name in enumerate(model_dict):
        # Build/load network
        config_path = config_path_prefix + model_dict[model_name]["config"]
        network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)

        # Check if network data is already saved in plot_data file
        root_dir = ut.get_project_root()
        if os.path.exists(root_dir+'/data/.plot_data.h5'):
            with h5py.File(root_dir+'/data/.plot_data.h5', 'r') as f:
                if (network.name not in f) or (overwrite==True):
                    # Load network (and use to rerun analyses)
                    saved_network_path = saved_network_path_prefix + model_dict[model_name]["pickle"]
                    try:
                        network.load(saved_network_path)
                    except:
                        ut.rename_population(network, 'I', 'SomaI')
                        network.load(saved_network_path)


        # Plot 1: Average population activity / batch accuracy to the test dataset
        ax = fig.add_subplot(axes[0, i])
        pt.plot_batch_accuracy(network, test_dataloader, population=network.Output.E, ax=ax, export=True, overwrite=False)
        ax.set_title(model_dict[model_name]["name"], fontsize=10)
        if i>0:
            ax.set_ylabel('')


        # Plot 2: Example receptive fields
        population = network.H1.E
        num_units = 9
        rf_axes = gs.GridSpecFromSubplotSpec(3, 3, subplot_spec=axes[1, i], wspace=0.1, hspace=-0.3)

        ax_list = []
        for j in range(num_units):
            ax = fig.add_subplot(rf_axes[j])
            ax_list.append(ax)
        receptive_fields = ut.compute_maxact_receptive_fields(population, test_dataloader, export=True, overwrite=False)
        im = pt.plot_receptive_fields(receptive_fields, sort=True, ax_list=ax_list)

        fig_width, fig_height = fig.get_size_inches()
        cax = fig.add_axes([0.06, ax.get_position().y0-0.1/fig_height, 0.1, 0.05/fig_height])
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')

        # Plot 3:
        ax = fig.add_subplot(axes[2, i])

        # Plot 4: Learning curves

        
        # # Compute test activity
        # plot_data = {}
        # output, labels = ut.compute_test_activity(network, test_dataloader, sorted_output_idx=None)
        # average_activity = ut.compute_average_activity(output, labels)
        # plot_data[network.name] = {'average_activity_OutputE': average_activity)


        # if network.name in plot_data:
        #     average_output_activity = plot_data[network.name]['average_activity_OutputE']

        #     ax = fig.add_subplot(axes[0, i])
        #     pt.plot_average_population_activity('Output', average_output_activity, ax)

    if show:
        plt.show()

    if save:
        fig.savefig("figures/Figure1.png", dpi=300)
        fig.savefig("figures/Figure1.svg", dpi=300)




if __name__=="__main__":
    pt.update_plot_defaults()

    all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)

    generate_Figure1(all_dataloaders)