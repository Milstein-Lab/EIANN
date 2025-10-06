import click
from time import time
import EIANN.utils as ut

@click.command()
@click.option('--network-config-file-name')
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data/mnist')
def main(network_config_file_name, data_dir):
    start_time = time()

    # Load dataset
    train_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_MNIST_dataloaders(data_dir=data_dir)

    network_seed = 66049
    data_seed = 257

    ut.set_all_seeds(seed=network_seed)

    # Create network object
    config_file_path = f"EIANN/network_config/mnist/{network_config_file_name}"
    network = ut.build_EIANN_from_config(config_file_path, network_seed=network_seed)

    # Train network
    data_generator.manual_seed(data_seed)
    network.train(train_dataloader, val_dataloader, 
                epochs = 1,
                samples_per_epoch = 20_000, 
                val_interval = (0, -1, 100), 
                store_history = False,
                store_history_interval = (0, -1, 100), 
                store_dynamics = False, 
                store_params = False,
                status_bar = False)
    
    network.run_time = time() - start_time

    print(f'Using Device: {network.device}')
    print(f'Network Name: {network_config_file_name}')
    print(f'Final Val Accuracy: {network.val_accuracy_history[-1]}')
    print(f'Final Val Loss: {network.val_loss_history[-1]}')
    print(f'Network Run Time: {network.run_time} sec')

    print(f"Network Sample Order: {network.sample_order}")

if __name__ == '__main__':
    main()



######## Network Tracking (Bridges) ########

# ------ CPU ------

# Van BP 
# - Network Name: 20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 96.55999755859375
# - Final Val Loss: 0.008532393723726273
# - Network Run Time: 80.56694316864014 sec

# BP Dale
# - Network Name: 20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 95.86000061035156
# - Final Val Loss: 0.010352011770009995
# - Network Run Time: 1154.7670638561249 sec


# ------ GPU ------

# Van BP 
# - Network Name: 20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 96.30999755859375
# - Final Val Loss: 0.008542143739759922
# - Network Run Time: 39.9001362323761 sec

# BP Dale 
# - Network Name: 20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 95.22999572753906
# - Final Val Loss: 0.01068859826773405
# - Network Run Time: 306.1142373085022 sec
