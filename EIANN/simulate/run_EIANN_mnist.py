import click
import torch
from time import time
import EIANN.utils as ut

@click.command()
@click.option('--network-config-file-name', help="YAML file in EIANN/network_config/mnist", required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data/mnist')
@click.option('--debug', default=False, is_flag=True, help="Enable debug mode")
def main(network_config_file_name, data_dir, debug):
    start_time = time()

    network_seed = 66049
    data_seed = 257

    # Determine device FIRST
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ut.set_all_seeds(seed=network_seed)

    print(f"1. Random num: {torch.rand(1)}")

    # Load dataset
    train_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_MNIST_dataloaders(data_dir=data_dir)
    if debug:
        print("Loaded Data")

    print(f"2. Random num: {torch.rand(1)}")

    # Create network object
    config_file_path = f"EIANN/network_config/mnist/{network_config_file_name}"
    network = ut.build_EIANN_from_config(config_file_path, network_seed=network_seed, device=device)
    if debug:
        print("Built Network")

    print(f"3. Random num: {torch.rand(1)}")

    print('Weights before train')
    print(network.H1.E.Input.E.weight.detach().cpu()[0,0:10])

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
    if debug:
        print("Trained Network")

    print(f"4. Random num: {torch.rand(1)}")
    
    network.run_time = time() - start_time

    print(f'Using Device: {network.device}')
    print(f'Network Name: {network_config_file_name}')
    print(f'Final Val Accuracy: {network.val_accuracy_history[-1]}')
    print(f'Final Val Loss: {network.val_loss_history[-1]}')
    print(f'Network Run Time: {network.run_time} sec')

    print('Weights after train')
    print(network.H1.E.Input.E.weight.detach().cpu()[0,0:10])

    print(f"Network Sample Order: {network.sample_order}")

if __name__ == '__main__':
    main()

# TODO compare random numbers for GPU and CPU in .o file for jobs 35418474 (gpu) and 35418477 (cpu) 
# TODO raytune optimizer


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

# BP_like_5J
# - Network Name: 20241009_EIANN_2_hidden_mnist_BP_like_config_5J_complete_optimized.yaml
# - Final Val Accuracy: 93.4000015258789
# - Final Val Loss: 0.017700176686048508
# - Network Run Time: 1234.9985365867615 sec

# Hebb_Temp_Contrast (job id: 35403211)
# - Network Name: 20241125_EIANN_2_hidden_mnist_Hebb_Temp_Contrast_config_2_complete_optimized.yaml
# - Final Val Accuracy: 91.9800033569336
# - Final Val Loss: 0.03151409327983856
# - Network Run Time: 1423.9688520431519 sec


# ------ GPU ------

# Van BP 
# - Network Name: 20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 96.30999755859375
# - Final Val Loss: 0.008542143739759922
# - Network Run Time: 39.9001362323761 sec

# Van BP (with AMP) (job id: 35403673)
# - Network Name: 20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 96.30999755859375
# - Final Val Loss: 0.008542143739759922
# - Network Run Time: 46.637263774871826 sec

# BP Dale 
# - Network Name: 20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 95.22999572753906
# - Final Val Loss: 0.01068859826773405
# - Network Run Time: 306.1142373085022 sec

# BP_like_5J
# - Network Name: 20241009_EIANN_2_hidden_mnist_BP_like_config_5J_complete_optimized.yaml
# - Final Val Accuracy: 93.47999572753906
# - Final Val Loss: 0.01799418218433857
# - Network Run Time: 362.48494958877563 sec

# Hebb_Temp_Contrast (job id: 35403210)
# - Network Name: 20241125_EIANN_2_hidden_mnist_Hebb_Temp_Contrast_config_2_complete_optimized.yaml
# - Final Val Accuracy: 92.30999755859375
# - Final Val Loss: 0.03272122144699097
# - Network Run Time: 727.7840685844421 sec