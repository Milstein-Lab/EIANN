import pytest
import os
import EIANN as eiann
import EIANN.utils as ut
import EIANN.plot as pt
import EIANN.network as nt


def test_build_network(root_dir):
    config_dir = f"{root_dir}/EIANN/network_config/mnist/"
    assert len(os.listdir(config_dir)) > 0, "No network configs found in the network config dir"
    
    for config_name in os.listdir(config_dir):
        if config_name.endswith(".yaml") or config_name.endswith(".yml"):
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


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Pickle file not saved in GitHub Actions")
def test_load_network(network, root_dir, dataloaders_mnist):
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator =  dataloaders_mnist    
    network_name = "20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized"
    network_seed = 66049
    data_seed = 257
    saved_network_path = root_dir + f"/EIANN/data/saved_network_pickles/mnist/{network_name}_{network_seed}_{data_seed}.pkl"
    network = ut.load_network(saved_network_path)
    idx, data, target = next(iter(train_dataloader))
    network.forward(data, no_grad=True)