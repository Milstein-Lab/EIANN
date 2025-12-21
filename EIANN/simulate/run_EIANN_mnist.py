import click
import torch
from time import time
import EIANN.utils as ut

@click.command()
@click.option('--network-config-file-name', help="YAML file in EIANN/network_config/mnist", required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data/mnist')
@click.option('--debug', default=False, is_flag=True, help="Enable debug mode")
@click.option('--network-seed', type=int, default=None, help="Seed for network initialization")
@click.option('--device', type=str, default='cpu', help="Device to use: 'cuda' or 'cpu'")
def main(network_config_file_name, data_dir, debug, network_seed, device):
    start_time = time()

    seed_map = {
        66049: 257,
        66050: 258,
        66051: 259,
        66052: 260,
        66053: 261,
    }

    if network_seed is not None and network_seed in seed_map.keys():
        data_seed = seed_map[network_seed]
    else:
        network_seed = 66049
        data_seed = 257

    print(f'Network seed: {network_seed}')
    print(f'Data seed: {data_seed}')

    ut.set_all_seeds(seed=network_seed)

    # Load dataset
    train_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_MNIST_dataloaders(data_dir=data_dir)

    # Create network
    config_file_path = f"EIANN/network_config/mnist/{network_config_file_name}"
    network = ut.build_EIANN_from_config(config_file_path, network_seed=network_seed, device=device)

    if debug:
        print('Weights before train')
        weights_before = network.module_dict['H1E_InputE'].weight.detach().clone().cpu()[0, 350:360]
        print(weights_before)
        state_dict_before = {k: v.detach().clone().cpu() for k, v in network.state_dict().items()}

    # Train network
    data_generator.manual_seed(data_seed)
    network.train(train_dataloader, val_dataloader, 
                epochs=1,
                samples_per_epoch=20_000, 
                val_interval=(0, -1, 100), 
                store_history=False,
                store_history_interval=(0, -1, 100), 
                store_dynamics=False, 
                store_params=False,
                status_bar=False)

    network.run_time = time() - start_time

    print(f'\nUsing Device: {network.device}')
    print(f'Network Name: {network_config_file_name}')
    print(f'Final Val Accuracy: {network.val_accuracy_history[-1]}')
    print(f'Final Val Loss: {network.val_loss_history[-1]}')
    print(f'Network Run Time: {network.run_time} sec')

    if debug:
        print('\nWeights after train')
        weights_after = network.module_dict['H1E_InputE'].weight.detach().clone().cpu()[0, 350:360]
        print(weights_after)
        state_dict_after = {k: v.detach().clone().cpu() for k, v in network.state_dict().items()}
    
        print('\nWeight difference:')
        weight_diff = weights_after - weights_before
        print(weight_diff)
        print(f'Max absolute change: {weight_diff.abs().max().item():.6f}')
        print(f'Mean absolute change: {weight_diff.abs().mean().item():.6f}')

        for key in state_dict_before:
            diff = (state_dict_after[key] - state_dict_before[key]).abs()
            print(f"{key}: max_change={diff.max().item():.6f}, mean_change={diff.mean().item():.6f}")

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

# BP_like_5J
# - Network Name: 20241009_EIANN_2_hidden_mnist_BP_like_config_5J_complete_optimized.yaml
# - Final Val Accuracy: 93.4000015258789
# - Final Val Loss: 0.017700176686048508
# - Network Run Time: 1234.9985365867615 sec

# Hebb_Temp_Contrast
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

# Van BP (with AMP)
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

# Hebb_Temp_Contrast
# - Network Name: 20241125_EIANN_2_hidden_mnist_Hebb_Temp_Contrast_config_2_complete_optimized.yaml
# - Final Val Accuracy: 92.30999755859375
# - Final Val Loss: 0.03272122144699097
# - Network Run Time: 727.7840685844421 sec