import pytest
import torch
import os

import EIANN.utils as ut
import EIANN.plot as pt
import EIANN._network as nt


def test_build_network(root_dir):
    config_dir = root_dir + "/EIANN/network_config/mnist/"
    assert len(os.listdir(config_dir)) > 0, "No network configs found in the network config dir"
    assert all([(config_name.endswith(".yaml") or config_name.endswith(".yml")) for config_name in os.listdir(config_dir)]), "Not all files in the network config dir are yaml files"

    for config_name in os.listdir(config_dir):
        try:
            network = ut.build_EIANN_from_config(config_dir + config_name, network_seed=66049)
        except:
            raise Exception(f"Failed to build network from config: {config_name}")
        assert isinstance(network, nt.Network)


def test_train_network(network, dataloaders_mnist):
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator =  dataloaders_mnist
    network.train(train_sub_dataloader, 
                  test_dataloader, 
                  epochs=1,
                  val_interval=(0,-1,1),
                  store_history=True, 
                  store_params=True,
                  status_bar=True)
    ut.save_network(network, "test_network.pkl")
    os.remove("test_network.pkl")
    ut.save_network(network, "test_dir/test_network.pkl")
    os.remove("test_dir/test_network.pkl")
    os.rmdir("test_dir")

# @pytest.mark.skipif(os.environ.get("CI") == "true", reason="Pickle file not saved in GitHub Actions")
# def test_load_network(network, root_dir):
#     saved_network_dir = root_dir + "/EIANN/data/mnist/"
#     saved_network = "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl"
#     saved_network_path = saved_network_dir + saved_network
#     network = ut.load_network(saved_network_path)