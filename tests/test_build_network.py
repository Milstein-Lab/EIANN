import pytest
import torch
import numpy as np

import matplotlib.pyplot as plt

import EIANN.utils as ut
import EIANN.plot as pt
import EIANN._network as nt

@pytest.fixture(scope="module")
def network():
    root_dir = ut.get_project_root()
    config_dir = "/EIANN/network_config/mnist/"
    config_name = "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml"
    test_config = root_dir + config_dir + config_name
    return ut.build_EIANN_from_config(test_config, network_seed=66049)

@pytest.fixture(scope="module")
def MNIST_dataloaders():
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_MNIST_dataloaders(sub_dataloader_size=1)
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(train_sub_dataloader, torch.utils.data.DataLoader)
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)
    assert isinstance(test_dataloader, torch.utils.data.DataLoader)
    assert isinstance(data_generator, torch.Generator)
    return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator

def test_train_network(network, MNIST_dataloaders):
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator =  MNIST_dataloaders
    network.train(train_sub_dataloader, 
                  test_dataloader, 
                  epochs=1,
                  val_interval=(0,-1,1),
                  store_history=True, 
                  store_params=True,
                  status_bar=True)

# def test_load_network():
#     root_dir = ut.get_project_root()
#     config_dir = "/EIANN/network_config/mnist/"
#     config_name = "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml"
#     test_config = root_dir + config_dir + config_name

#     saved_network_dir = "/EIANN/data/mnist/"
#     saved_network = "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_66049_257.pkl"
#     saved_network_path = root_dir + saved_network_dir + saved_network

#     print(test_config)
#     print(saved_network_path)

#     bpDale_network = ut.build_EIANN_from_config(test_config, network_seed=66049)
#     bpDale_network.load(saved_network_path)