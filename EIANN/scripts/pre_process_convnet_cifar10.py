import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import click
import h5py

import EIANN.utils as utils



@click.command()
@click.option("--data-file-path", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../data/cifar10/20250812_EIANN_2_hidden_convnet_cifar10_van_bp_relu_SGD_MSE_config_G_zero_bias_66049_257_10_epochs.pkl')
@click.option("--population", type=str, default='Conv2FlatE')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data/cifar10')
@click.option("--interactive", is_flag=True)
@click.option("--disp", is_flag=True)
def main(data_file_path, population, output_dir, interactive, disp):
    """
    :param data_file_path: str (path to .pkl file containing exported trained Network)
    :param population: str
    :param output_dir: str (path to dir)
    :param interactive: bool
    :param disp: bool
    """
    export_file_path = data_file_path.rsplit('.', 1)[0] + '_preprocessed_data.h5'
    
    # Load trained network
    network = utils.load_network(data_file_path, disp)
    
    # Load dataset
    if interactive:
        download = True
    else:
        download = False
    
    tensor_transform = T.ToTensor()
    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(root=output_dir + '/datasets/CIFAR10_data/',
                                                         train=True, download=download, transform=tensor_transform)
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(root=output_dir + '/datasets/CIFAR10_data/',
                                                        train=False, download=download, transform=tensor_transform)
    
    # Add index to train & test data
    CIFAR10_train = []
    for idx, (data, target) in enumerate(CIFAR10_train_dataset):
        target = torch.eye(len(CIFAR10_train_dataset.classes))[target]
        CIFAR10_train.append((idx, data, target))
    
    CIFAR10_test = []
    for idx, (data, target) in enumerate(CIFAR10_test_dataset):
        target = torch.eye(len(CIFAR10_test_dataset.classes))[target]
        CIFAR10_test.append((idx, data, target))
    
    # Put data in dataloader
    train_sub_dataloader = \
        torch.utils.data.DataLoader(CIFAR10_train[0:-10000], batch_size=len(CIFAR10_train)-10000, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(CIFAR10_train[-10000:], batch_size=10000, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=10000, shuffle=False)
    
    idx, data, target = next(iter(train_sub_dataloader))
    network.forward(data, no_grad=True)
    with h5py.File(export_file_path, 'w') as f:
        group = f.create_group('train_sub_data')
        group.create_dataset('idx', data=idx.numpy())
        group.create_dataset('data', data=network.populations[population].activity.detach().numpy())
        group.create_dataset('target', data=target.numpy())
    
    idx, data, target = next(iter(val_dataloader))
    network.forward(data, no_grad=True)
    with h5py.File(export_file_path, 'a') as f:
        group = f.create_group('val_data')
        group.create_dataset('idx', data=idx.numpy())
        group.create_dataset('data', data=network.populations[population].activity.detach().numpy())
        group.create_dataset('target', data=target.numpy())

    idx, data, target = next(iter(test_dataloader))
    network.forward(data, no_grad=True)
    with h5py.File(export_file_path, 'a') as f:
        group = f.create_group('test_data')
        group.create_dataset('idx', data=idx.numpy())
        group.create_dataset('data', data=network.populations[population].activity.detach().numpy())
        group.create_dataset('target', data=target.numpy())
    
    if disp:
        print('Exported preprocessed data to %s' % export_file_path)
    
    if interactive:
        globals().update(locals())
    
if __name__ == '__main__':
    main(standalone_mode=False)