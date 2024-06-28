import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import os
import h5py

import EIANN.utils as ut
import EIANN.plot as pt

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

def generate_Fig1(all_dataloaders, show=True, save=False):
    '''
    Compare vanilla Backprop to networks with 'cortical' architecures (i.e. with somatic feedback inhibition). 
    All networks have 1 hidden layer with 500 E units and 50 somaI units.
    '''

    model_dict = {"vanBP":      {"config": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl",
                                 "color": "black",
                                 "name": "Vanilla BP"},
                  "bpDale":     {"config": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_66049_257.pkl",
                                 "color": "green",
                                 "name": "Backprop with Dale's Law"},
                 }

    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 200 * mm))
    axes = gs.GridSpec(nrows=4, ncols=3,
                    left=0.06,right=0.9,
                    top=0.96, bottom = 0.04,
                    wspace=0.3, hspace=0.5)
    
    axes_metrics = gs.GridSpec(nrows=6, ncols=3,
                left=0.06,right=0.9,
                top=0.96, bottom = 0.04,
                wspace=0.3, hspace=0.5)
    ax_loss = fig.add_subplot(axes_metrics[0, 2])
    ax_accuracy = fig.add_subplot(axes_metrics[1, 2])
    ax_H1E_sparsity = fig.add_subplot(axes_metrics[2, 2])
    ax_H1E_selectivity = fig.add_subplot(axes_metrics[3, 2])
    ax_H1E_structure = fig.add_subplot(axes_metrics[4, 2])
    ax_H1E_discriminability = fig.add_subplot(axes_metrics[5, 2])

    load = True
    overwrite = False

    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    for i,model_name in enumerate(model_dict):
        # Build network model
        config_path_prefix = "network_config/mnist/"
        config_path = config_path_prefix + model_dict[model_name]["config"]
        network_seed = 66049
        network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)
        
        if load:
            # Load network from pickle (and use to rerun analyses)
            saved_network_path_prefix = "data/mnist/"
            saved_network_path = saved_network_path_prefix + model_dict[model_name]["pickle"]
            try:
                network.load(saved_network_path)
            except:
                ut.rename_population(network, 'I', 'SomaI')
                network.load(saved_network_path)


        # Plot 1: Average population activity of OutputE (batch accuracy to the test dataset)
        ax = fig.add_subplot(axes[i*2, 0])
        pt.plot_batch_accuracy(network, test_dataloader, population=network.Output.E, ax=ax, export=True, overwrite=overwrite)
        ax.set_title(model_dict[model_name]["name"], fontsize=10)


        # Plot 2: Example output receptive fields
        population = network.Output.E
        num_units = 10
        ax = fig.add_subplot(axes[i*2, 1])
        ax.axis('off')
        if i == 0:
            ax.set_title("Receptive fields", fontsize=10)

        pos = ax.get_position()
        new_left = pos.x0 - 0.02  # Move left boundary to the left
        new_bottom = pos.y0 + 0.06 # Move bottom boundary up
        new_height = pos.height - 0.09  # Decrease height
        ax.set_position([new_left, new_bottom, pos.width, new_height])
        rf_axes = gs.GridSpecFromSubplotSpec(2, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)

        ax_list = []
        for j in range(num_units):
            ax = fig.add_subplot(rf_axes[j])
            ax_list.append(ax)
            ax.axis('off')
        
        receptive_fields = ut.compute_maxact_receptive_fields(population, export=True, overwrite=overwrite)
        im = pt.plot_receptive_fields(receptive_fields, sort=False, ax_list=ax_list)
        fig_width, fig_height = fig.get_size_inches()
        # cax = fig.add_axes([0.03, ax.get_position().y0-0.1/fig_height, 0.1, 0.05/fig_height])

        cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                            0.1, 0.05/fig_height])
        fig.colorbar(im, cax=cax, orientation='horizontal')


        # Plot 3: Average population activity of H1E
        ax = fig.add_subplot(axes[1+i*2, 0])
        pt.plot_batch_accuracy(network, test_dataloader, population=network.H1.E, ax=ax, export=True, overwrite=overwrite)


        # Plot 4: Example hidden receptive fields
        population = network.H1.E
        num_units = 20
        ax = fig.add_subplot(axes[1+i*2, 1])
        ax.axis('off')

        pos = ax.get_position()
        # new_bottom = pos.y0 + 0.0  # Move bottom boundary up
        # new_width = pos.width + 0.04  # Increase width
        # new_height = pos.height - 0.01  # Decrease height

        new_left = pos.x0 - 0.02  # Move left boundary to the left
        new_bottom = pos.y0
        new_height = pos.height

        ax.set_position([new_left, new_bottom, pos.width, new_height])
        rf_axes = gs.GridSpecFromSubplotSpec(4, 5, subplot_spec=ax, wspace=0.1, hspace=0.1)

        ax_list = []
        for j in range(num_units):
            ax = fig.add_subplot(rf_axes[j])
            ax_list.append(ax)
            ax.axis('off')
        
        receptive_fields_H1E = ut.compute_maxact_receptive_fields(population, export=True, overwrite=overwrite)
        im = pt.plot_receptive_fields(receptive_fields_H1E, sort=True, ax_list=ax_list)
        fig_width, fig_height = fig.get_size_inches()
        cax = fig.add_axes([ax_list[0].get_position().x0, ax.get_position().y0-0.1/fig_height, 
                            0.1, 0.05/fig_height])
        fig.colorbar(im, cax=cax, orientation='horizontal')


        # Plot 4: Learning curves / metrics
        population = network.H1.E
        metrics_dict = ut.compute_representation_metrics(population, test_dataloader, receptive_fields_H1E)
        # metrics_dict = load_metrics(model_dict[model_name]["pickle"], overwrite)

        # ax_H1E_selectivity.plot(metrics_dict["selectivity"], label=model_dict[model_name]["name"])
        # ax_H1E_discriminability.plot(metrics_dict["discriminability"], label=model_dict[model_name]["name"])
        # ax_H1E_structure.plot(metrics_dict["structure"], label=model_dict[model_name]["name"])
        # ax_H1E_sparsity.plot(metrics_dict["sparsity"], label=model_dict[model_name]["name"])

        ax_H1E_sparsity.hist(metrics_dict['sparsity'],50)
        ax_H1E_sparsity.set_title('Sparsity distribution')
        ax_H1E_sparsity.set_ylabel('num patterns')
        ax_H1E_sparsity.set_xlabel('(1 - fraction active units)')

        ax_H1E_selectivity.hist(metrics_dict['selectivity'],50)
        ax_H1E_selectivity.set_title('Selectivity distribution')
        ax_H1E_selectivity.set_ylabel('num units')
        ax_H1E_selectivity.set_xlabel('(1 - fraction active patterns)')

        ax_H1E_discriminability.set_title('Discriminability distribution')
        ax_H1E_discriminability.hist(metrics_dict['discriminability'], 50)
        ax_H1E_discriminability.set_ylabel('pattern pairs')
        ax_H1E_discriminability.set_xlabel('(1 - cosine similarity)')

        if receptive_fields is not None:
            ax_H1E_structure.hist(metrics_dict['structure'], 50)
            ax_H1E_structure.set_title('Structure')
            ax_H1E_structure.set_ylabel('num units')
            ax_H1E_structure.set_xlabel('(1 - similarity to random noise)')
        else:
            ax_H1E_structure.axis('off')


        # ax = fig.add_subplot(axes[0, 2])
        # ax.plot(network.loss_history)
        # ax.set_ylabel('Train loss')
        # ax.set_xlabel('Training steps')


    # if show:
    #     plt.show()

    if save:
        fig.savefig("figures/Fig1.png", dpi=300)
        fig.savefig("figures/Fig1.svg", dpi=300)




if __name__=="__main__":
    # pt.update_plot_defaults()

    all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)

    generate_Fig1(all_dataloaders, save=True)