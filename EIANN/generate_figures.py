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
    
    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 180 * mm))
    axes = gs.GridSpec(nrows=4, ncols=4,
                       left=0.05,right=0.98,
                       top=0.95, bottom = 0.08,
                       wspace=0.4, hspace=0.4)
    
    model_dict = {"vanBP":      {"config": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl"},
                  "unsup_Hebb": {"config": "20231025_EIANN_1_hidden_mnist_Gjorgjieva_Hebb_config_F_optimized.yaml", 
                                 "pickle": "20230712_EIANN_1_hidden_mnist_Gjorgjieva_Hebb_config_F_66049_257.pkl"},
                  "bpDale":     {"config": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_optimized.yaml", 
                                 "pickle": "20231018_EIANN_1_hidden_mnist_bpDale_relu_SGD_config_G_66049_257.pkl"},
                  "sup_Hebb":   {"config": "20231025_EIANN_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_config_F_optimized.yaml", 
                                 "pickle": "20230505_EIANN_1_hidden_mnist_Supervised_Gjorgjieva_Hebb_config_F_66049_257.pkl"},
                 }


    config_path_prefix = "network_config/mnist/"
    saved_network_path_prefix = "data/mnist/"
    network_seed = 66049

    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = all_dataloaders

    for i, model in enumerate(model_dict):
        # Build/load network
        config_path = config_path_prefix + model_dict[model]["config"]
        saved_network_path = saved_network_path_prefix + model_dict[model]["pickle"]
        network = ut.build_EIANN_from_config(config_path, network_seed=network_seed)

        load_network = True
        if os.path.exists('data/.plot_data.h5'): # If figures have already been generated, use existing plot data instead of reloading the network
            with h5py.File('data/.plot_data.h5', 'r') as f:
                if network.name in f:
                    print("HDF5 file detected, loading plot data")
                    load_network = False

        if load_network:
            try:
                network.load(saved_network_path)
            except:
                ut.rename_population(network, 'I', 'SomaI')
                network.load(saved_network_path)
        
        # Plot accuracy
        ax = fig.add_subplot(axes[0, i])
        pt.plot_batch_accuracy(network, test_dataloader, ax=ax, use_hdf5=True)

    if show:
        plt.show()

    if save:
        fig.savefig("figures/Figure1.png", dpi=300)
        fig.savefig("figures/Figure1.svg", dpi=300)




if __name__=="__main__":
    pt.update_plot_defaults()

    all_dataloaders = ut.get_MNIST_dataloaders(sub_dataloader_size=1000)

    generate_Figure1(all_dataloaders)