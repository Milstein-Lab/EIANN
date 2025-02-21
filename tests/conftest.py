import pytest
import torch
import EIANN.utils as ut


@pytest.fixture(scope="module") #Make the output of this function available to all tests in this module
def dataloaders_mnist():
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_MNIST_dataloaders(sub_dataloader_size=1)
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(train_sub_dataloader, torch.utils.data.DataLoader)
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)
    assert isinstance(test_dataloader, torch.utils.data.DataLoader)
    assert isinstance(data_generator, torch.Generator)
    return train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator


@pytest.fixture(scope="module") 
def dataloaders_spiral():
    train_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_spiral_dataloaders(batch_size='full_dataset')
    return train_dataloader, val_dataloader, test_dataloader, data_generator


@pytest.fixture(scope="module") 
def root_dir():
    root_dir = ut.get_project_root()
    return root_dir


@pytest.fixture(scope="module")
def network(root_dir):
    config_dir = "/EIANN/network_config/mnist/"
    config_name = "20231120_EIANN_1_hidden_mnist_van_bp_relu_SGD_config_G_optimized.yaml"
    test_config = root_dir + config_dir + config_name
    network = ut.build_EIANN_from_config(test_config, network_seed=66049)
    return network