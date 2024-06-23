import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import h5py

import EIANN.utils as ut
import EIANN.plot as pt

'''
Figure 1: Van_BP vs bpDale(learnedI)
        -> bpDale is more structured/sparse (focus on H1E metrics)

Figure 2: bpDale(learnedI) vs top-sup HebbWN(learnedI) vs unsup-HebbWN(learnedI)
        (top: Output+H1 E, activity and receptive fields and metrics)
        (bottom: Output+H1 SomaI, activity and metrics)
        -> bpDale SomaI is unselective/unstructured, biological learning rule is not (focus on SomaI metrics)
        -> sup HebbWN performance is mediocre (representational collapse), we need a biological way to pass gradients to hidden layers

(switched to fixed somaI)
-> bpDale still performs with fixedI, HebbWN does not, but BCM does (focus on H1E metrics)
(Supp: bpDale(fixedI) vs top-sup HebbWN(fixedI) vs top-sup BCM(fixedI) vs unsup BCM(fixedI))

Passing gradients with apical dendrites:

Figure 3: dendritic_gating(bpLike) vs dendritic_gating w/ HebbWN dendritic learning
        -> Can compute local loss with top-down nudges and dendI subtraction
        -> dendritic subtraction is effective even with HebbWN

Figure 4: BTSP(weight transpose + HebbWN dendI) vs sup-HebbWN
        -> Local loss allows learning with a biological rule
        -> BTSP does better than Hebbian (Hebb too simple)

Figure 5: BTSP(learned top-down W + HebbWN dendI)
'''

def generate_model_comparison_figure(all_dataloaders, show=True, save=False):
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
        


    model_dict = {"vanBP":      {"config": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl",
                                 "color": "black",
                                 "name": "Vanilla BP"},
                  "bpDale":     {"config": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_66049_257.pkl",
                                 "color": "green",
                                 "name": "Backprop with Dale's Law"},
                #   "unsup_Hebb": {"config": "20231025_EIANN_1_hidden_mnist_Gjorgjieva_Hebb_config_F_optimized.yaml", 
                #                  "pickle": "20230712_EIANN_1_hidden_mnist_Gjorgjieva_Hebb_config_F_66049_257.pkl",
                #                  "color": "blue",
                #                  "name": "Unsupervised Hebb"},
                #   "sup_Hebb":   {"config": "20231025_EIANN_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_config_F_optimized.yaml", 
                #                  "pickle": "20230505_EIANN_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_config_F_66049_257.pkl",
                #                  "color": "red",
                #                  "name": "Supervised Hebb"},
                 }

    config_path_prefix = "network_config/mnist/"
    saved_network_path_prefix = "data/mnist/"
    network_seed = 66049
    overwrite = True

    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    for model_name in model_dict:
        mm = 1 / 25.4  # millimeters in inches
        fig = plt.figure(figsize=(180 * mm, 180 * mm))
        axes = gs.GridSpec(nrows=4, ncols=4,
                        left=0.06,right=0.98,
                        top=0.96, bottom = 0.04,
                        wspace=0.4, hspace=0.2)

        # Build/load network
        config_path = config_path_prefix + model_dict[model_name]["config"]
        network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)

        # Check if network data is already saved in plot_data file
        root_dir = ut.get_project_root()
        if os.path.exists(root_dir+'EIANN/data/.plot_data.h5'):
            with h5py.File(root_dir+'EIANN/data/.plot_data.h5', 'r') as f:
                if (network.name not in f) or (overwrite==True):
                    # Load network (and use to rerun analyses)
                    saved_network_path = saved_network_path_prefix + model_dict[model_name]["pickle"]
                    try:
                        network.load(saved_network_path)
                    except:
                        ut.rename_population(network, 'I', 'SomaI')
                        network.load(saved_network_path)


        # Plot 1: Average population activity of OutputE (batch accuracy to the test dataset)
        ax = fig.add_subplot(axes[0, 0])
        pt.plot_batch_accuracy(network, test_dataloader, population=network.Output.E, ax=ax, export=True, overwrite=False)
        ax.set_title(model_dict[model_name]["name"], fontsize=10)


        # Plot 2: Example output receptive fields
        population = network.Output.E
        num_units = 10
        ax = fig.add_subplot(axes[0, 1])
        ax.axis('off')

        pos = ax.get_position()
        new_left = pos.x0 - 0.03  # Move left boundary to the left
        new_bottom = pos.y0 + 0.08  # Move bottom boundary up
        new_width = pos.width + 0.04  # Increase width
        new_height = pos.height - 0.1  # Decrease height
        ax.set_position([new_left, new_bottom, new_width, new_height])
        rf_axes = gs.GridSpecFromSubplotSpec(2, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)

        ax_list = []
        for j in range(num_units):
            ax = fig.add_subplot(rf_axes[j])
            ax_list.append(ax)
            ax.axis('off')
        
        receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, overwrite=False)
        im = pt.plot_receptive_fields(receptive_fields, sort=False, ax_list=ax_list)
        fig_width, fig_height = fig.get_size_inches()
        # cax = fig.add_axes([0.03, ax.get_position().y0-0.1/fig_height, 0.1, 0.05/fig_height])

        cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                            0.1, 0.05/fig_height])
        fig.colorbar(im, cax=cax, orientation='horizontal')


        # Plot 3: Average population activity of H1E
        ax = fig.add_subplot(axes[1, 0])
        pt.plot_batch_accuracy(network, test_dataloader, population=network.H1.E, ax=ax, export=True, overwrite=False)
        ax.set_title(model_dict[model_name]["name"], fontsize=10)

        # # Plot 4: Example hidden receptive fields
        population = network.H1.E
        num_units = 25
        ax = fig.add_subplot(axes[1, 1])
        ax.axis('off')

        pos = ax.get_position()
        new_left = pos.x0 - 0.03  # Move left boundary to the left
        new_bottom = pos.y0 + 0.0  # Move bottom boundary up
        new_width = pos.width + 0.04  # Increase width
        new_height = pos.height - 0.01  # Decrease height
        ax.set_position([new_left, new_bottom, new_width, new_height])
        rf_axes = gs.GridSpecFromSubplotSpec(5, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)

        ax_list = []
        for j in range(num_units):
            ax = fig.add_subplot(rf_axes[j])
            ax_list.append(ax)
            ax.axis('off')
        
        receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, overwrite=False)
        im = pt.plot_receptive_fields(receptive_fields, sort=True, ax_list=ax_list)
        fig_width, fig_height = fig.get_size_inches()
        cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                            0.1, 0.05/fig_height])
        fig.colorbar(im, cax=cax, orientation='horizontal')



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

    # if save:
    #     fig.savefig("figures/model_comparison.png", dpi=300)
    #     fig.savefig("figures/model_comparison.svg", dpi=300)




if __name__=="__main__":
    pt.update_plot_defaults()

    all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)

    generate_model_comparison_figure(all_dataloaders)