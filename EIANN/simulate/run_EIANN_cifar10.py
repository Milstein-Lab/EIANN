import click
import torch
import torchvision
import torchvision.transforms as T
from time import time
import EIANN.utils as ut

@click.command()
@click.option('--network-config-file-name', help="YAML file in EIANN/network_config/cifar10", required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data/cifar10')
@click.option('--debug', default=False, is_flag=True, help="Enable debug mode")
@click.option('--network-seed', type=int, default=None, help="Seed for network initialization")
@click.option('--device', type=str, default='cpu', help="Device to use: 'cuda' or 'cpu'")
def main(network_config_file_name, data_dir, debug, network_seed, device, flatten_data=False):
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
    if flatten_data:
        tensor_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(torch.flatten)])
    else:
        tensor_transform = T.ToTensor()

    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(root=data_dir + '/cifar10', train=True, 
                                                         download=True, transform=tensor_transform)
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(root=data_dir + '/cifar10', train=False, 
                                                        download=True, transform=tensor_transform)
    
    # Add index to train & test data
    CIFAR10_train = []

    for idx, (data, target) in enumerate(CIFAR10_train_dataset):
        target = torch.eye(len(CIFAR10_train_dataset.classes))[target]
        CIFAR10_train.append((idx, data, target))
    CIFAR10_val = CIFAR10_train[-10000:]
    CIFAR10_train = CIFAR10_train[:-10000]
    
    CIFAR10_test = []
    for idx, (data, target) in enumerate(CIFAR10_test_dataset):
        target = torch.eye(len(CIFAR10_test_dataset.classes))[target]
        CIFAR10_test.append((idx, data, target))
    
    # Put data in dataloader
    data_generator = torch.Generator()
    train_sub_dataloader = torch.utils.data.DataLoader(CIFAR10_train, shuffle=True, generator=data_generator)
    val_dataloader = torch.utils.data.DataLoader(CIFAR10_val, batch_size=10000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=10000, shuffle=False)

    # Create network
    config_file_path = f"EIANN/network_config/cifar10/{network_config_file_name}"
    network = ut.build_EIANN_from_config(config_file_path, network_seed=network_seed, device=device)

    if debug:
        print('Weights before train')
        weights_before = network.module_dict['H1E_InputE'].weight.detach().clone().cpu()[0, 350:360]
        print(weights_before)
        state_dict_before = {k: v.detach().clone().cpu() for k, v in network.state_dict().items()}

    # Train network
    data_generator.manual_seed(data_seed)
    network.train(train_sub_dataloader, val_dataloader, 
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

# Local run:
# python EIANN/simulate/run_EIANN_cifar10.py --network-config-file-name=20250812_EIANN_2_hidden_convnet_cifar10_van_bp_relu_SGD_CE_config_G_learned_bias.yaml --data-dir=EIANN/data/ --network-seed=66049 --device=cpu --debug
# Replace cpu with cuda for GPU run

# python -m nested.analyze --interactive --config-file-path=optimize/optimize_config/cifar10/20250814_nested_optimize_EIANN_2_hidden_convnet_cifar10_van_bp_relu_SGD_CE_config_G_learned_bias.yaml --disp --model-key=van_bp_CE_learned_bias --param-file-path=optimize/optimize_params/cifar10/20250815_nested_optimize_convnet_cifar10_params.yaml --output-dir=/data/cifar10 --status_bar

######## Network Tracking (Frontera) ########

