import click
from time import time
import EIANN.utils as ut

import torch, sys, platform, hashlib
print("python:", sys.version.splitlines()[0])
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())
print("cudnn deterministic:", torch.backends.cudnn.deterministic)
print("cudnn benchmark:", torch.backends.cudnn.benchmark)

@click.command()
@click.option('--network-config-file-name')
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data/mnist')
def main(network_config_file_name, data_dir):
    start_time = time()

    # network_config = read_from_yaml(network_config_file_path)
    # layer_config = network_config['layer_config']
    # projection_config = network_config['projection_config']
    # training_kwargs = network_config['training_kwargs']

    # Load dataset
    train_dataloader, val_dataloader, test_dataloader, data_generator = ut.get_MNIST_dataloaders(data_dir=data_dir)

    # After dataloader is created, print a checksum of first batch
    idx, data, target = next(iter(train_dataloader))
    s = hashlib.md5(data.flatten().cpu().numpy().tobytes()).hexdigest()
    print("first-train-batch-md5:", s)
    print("first-train-target-sum:", target.sum().item())

    network_seed = 66049
    data_seed = 257

    ut.set_all_seeds(seed=network_seed)

    # Create network object
    config_file_path = f"EIANN/network_config/mnist/{network_config_file_name}"
    network = ut.build_EIANN_from_config(config_file_path, network_seed=network_seed)

    # Small forward check on device before training
    net = network
    net.to(net.device)
    out = net.forward(data.to(net.device), no_grad=True).detach().cpu()
    print("forward output stats:", out.mean().item(), out.std().item())

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

    print(f'Network Name: {network_config_file_name}')
    print(f'Final Val Accuracy: {network.val_accuracy_history[-1]}')
    print(f'Final Val Loss: {network.val_loss_history[-1]}')
    print(f'Using Device: {network.device}')
    print(f'Network Run Time: {network.run_time} sec')

if __name__ == '__main__':
    main()


# Network Tracking

# Van BP (CPU)
# - Network Name: 20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 96.52999877929688
# - Final Val Loss: 0.008667627349495888
# - Network Run Time: 108.71107316017151 sec

# Van BP (GPU)
# - Network Name: 20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
# - Final Val Accuracy: 96.13999938964844
# - Final Val Loss: 0.008898369036614895
# - Network Run Time: 91.12481260299683 sec