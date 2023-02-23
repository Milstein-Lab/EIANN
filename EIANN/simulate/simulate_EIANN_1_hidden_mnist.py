import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os, sys, math
from copy import deepcopy
import numpy as np
import h5py
import click
import matplotlib.pyplot as plt

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics
from EIANN.plot import plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history, \
    evaluate_test_loss_history
from nested.utils import Context, param_array_to_dict, get_unknown_click_arg_dict
from nested.parallel import get_parallel_interface
from nested.optimize_utils import nested_parallel_init_contexts_interactive


context = Context()


def config_worker():
    context.seed_start = int(context.seed_start)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.data_seed_start = int(context.data_seed_start)
    context.epochs = int(context.epochs)
    context.status_bar = bool(context.status_bar)
    if 'debug' not in context():
        context.debug = False
    else:
        context.debug = bool(context.debug)
    if 'store_weights_interval' not in context():
        context.store_weights_interval = (0, -1, 10000)
    if 'equilibration_activity_tolerance' not in context():
        context.equilibration_activity_tolerance = 0.2
    else:
        context.equilibration_activity_tolerance = float(context.equilibration_activity_tolerance)

    context.train_steps = int(context.train_steps)

    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']

    # Load dataset
    tensor_flatten = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    MNIST_train_dataset = torchvision.datasets.MNIST(root=context.output_dir + '/datasets/MNIST_data/', train=True,
                                                     download=True, transform=tensor_flatten)
    MNIST_test_dataset = torchvision.datasets.MNIST(root=context.output_dir + '/datasets/MNIST_data/', train=False,
                                                    download=False, transform=tensor_flatten)

    # Add index to train & test data
    MNIST_train = []
    for idx, (data, target) in enumerate(MNIST_train_dataset):
        target = torch.eye(len(MNIST_train_dataset.classes))[target]
        MNIST_train.append((idx, data, target))

    MNIST_test = []
    for idx, (data, target) in enumerate(MNIST_test_dataset):
        target = torch.eye(len(MNIST_test_dataset.classes))[target]
        MNIST_test.append((idx, data, target))

    # Put data in dataloader
    context.data_generator = torch.Generator()
    context.train_sub_dataloader = \
        torch.utils.data.DataLoader(MNIST_train[0:context.train_steps], shuffle=True, generator=context.data_generator)
    context.val_dataloader = torch.utils.data.DataLoader(MNIST_train[-10000:], batch_size=10000, shuffle=False)
    context.test_dataloader = torch.utils.data.DataLoader(MNIST_test, batch_size=10000, shuffle=False)


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                     for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    if context.debug:
        print(network_seeds, data_seeds)
        sys.stdout.flush()
    return [network_seeds, data_seeds]


def simulate(seed, data_seed):
    """

    :param seed: int
    :param data_seed: int
    """
    plot = context.plot
    disp = context.disp
    debug = context.debug

    data_generator = context.data_generator
    train_sub_dataloader = context.train_sub_dataloader
    val_dataloader = context.val_dataloader
    test_dataloader = context.test_dataloader

    epochs = context.epochs

    network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)

    if plot:
        title = 'Initial (%i, %i)' % (seed, data_seed)
        plot_batch_accuracy(network, test_dataloader, population='all', title=title)

        for layer in network:
            for population in layer:
                for projection in population:
                    fig = plt.figure()
                    plt.imshow(projection.weight.data, aspect='auto', interpolation='none')
                    plt.suptitle('%s Weights: %s' % (title, projection.name))
                    plt.colorbar()
                    fig.show()

        for layer in network:
            for population in layer:
                fig = plt.figure()
                plt.imshow(torch.atleast_2d(population.activity.detach().T), aspect='auto', interpolation='none')
                plt.suptitle('%s Activity: %s' % (title, population.fullname))
                plt.colorbar()
                fig.show()

    if context.load_params or context.save_params:
        if context.data_file_name_base is None:
            network_config_file_name = context.network_config_file_path.rpartition('/')[-1]
            context.data_file_name_base = network_config_file_name.rpartition('.')[0]

        data_file_path = '%s/%s_%i_%i.pkl' % (context.output_dir, context.data_file_name_base, seed, data_seed)

    if context.load_params:
        if not os.path.isfile(data_file_path):
            raise Exception('simulate_EIANN_1_hidden_mnist: cannot load data from path: %s' % data_file_path)
        network.load(data_file_path)
    else:
        data_generator.manual_seed(data_seed)
        network.train_and_validate(train_sub_dataloader,
                                   val_dataloader,
                                   epochs=epochs,
                                   val_interval=context.val_interval, # e.g. (-201, -1, 10)
                                   store_history=True,
                                   store_weights=True,
                                   store_weights_interval=context.store_weights_interval,
                                   status_bar=context.status_bar)

    # reorder output units if using unsupervised/Hebbian rule
    if not context.supervised:
        min_loss_idx, sorted_output_idx = sort_by_val_history(network, plot=plot)
    else:
        min_loss_idx = torch.argmin(network.val_loss_history)
        sorted_output_idx = torch.arange(0, network.val_output_history.shape[-1])

    sorted_val_loss_history, sorted_val_accuracy_history = \
        recompute_validation_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, store=True, plot=plot)

    if context.save_params:
        network.save(data_file_path)

    _ = check_equilibration_dynamics(network, test_dataloader, context.equilibration_activity_tolerance, debug=True,
                                     disp=disp, plot=(debug and plot))

    if plot:
        title = 'Final (%i, %i)' % (seed, data_seed)
        plot_batch_accuracy(network, test_dataloader, population='all', sorted_output_idx=sorted_output_idx,
                            title=title)

        for layer in network:
            for population in layer:
                for projection in population:
                    fig = plt.figure()
                    plt.imshow(projection.weight.data, aspect='auto', interpolation='none')
                    plt.suptitle('%s Weights: %s' % (title, projection.name))
                    plt.colorbar()
                    fig.show()

        plot_train_loss_history(network)
        plot_validate_loss_history(network)
        evaluate_test_loss_history(network, test_dataloader, sorted_output_idx=sorted_output_idx, store_history=True,
                                   plot=True)


    # TODO: refactor
    # if export:
    #     if context.temp_output_path is not None:
    #
    #         end_index = start_index + context.num_training_steps_argmax_accuracy_window
    #         output_pop = network.output_pop
    #
    #         with h5py.File(context.temp_output_path, 'a') as f:
    #             if context.label is None:
    #                 label = str(len(f))
    #             else:
    #                 label = context.label
    #             group = f.create_group(label)
    #             model_group = group.create_group(str(seed))
    #             activity_group = model_group.create_group('activity')
    #             metrics_group = model_group.create_group('metrics')
    #             for post_layer in network:
    #                 post_layer_activity = activity_group.create_group(post_layer.name)
    #                 for post_pop in post_layer:
    #                     activity_data = \
    #                         post_pop.activity_history[network.sorted_sample_indexes, -1, :][start_index:end_index, :].T
    #                     if post_pop == output_pop:
    #                         activity_data = activity_data[sorted_output_idx,:]
    #                     post_layer_activity.create_dataset(post_pop.name, data=activity_data)
    #             metrics_group.create_dataset('loss', data=sorted_val_loss_history)
    #             metrics_group.create_dataset('accuracy', data=sorted_val_accuracy_history)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--network-config-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--data-file-name-base", type=str, default=None)
@click.option("--num-instances", type=int, default=1)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--load-params", is_flag=True)
@click.option("--save-params", is_flag=True)
@click.option("--label", type=str, default=None)
@click.option("--plot", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--disp", is_flag=True)
@click.option("--framework", type=str, default='serial')
@click.pass_context
def main(cli, config_file_path, network_config_file_path, data_file_name_base, num_instances, output_dir, export,
         export_file_path, load_params, save_params, label, plot, interactive, debug, disp, framework):
    """
    To execute on a single process:
    python -i simulate_EIANN_1_hidden_mnist.py --plot --interactive --config-file-path=$PATH_TO_CONFIG_YAML \
        --network-config-file-path=$PATH_TO_NETWORK_CONFIG_YAML

    To execute using MPI parallelism with 1 controller process and N - 1 worker processes:
    mpirun -n N python -i -m mpi4py.futures simulate_EIANN_1_hidden_mnist.py --plot --interactive --framework=mpi \
        --config-file-path=$PATH_TO_CONFIG_YAML --network-config-file-path=$PATH_TO_NETWORK_CONFIG_YAML

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path to .yaml file)
    :param network_config_file_path: str (path to .yaml file)
    :param data_file_name_base: str (path to .pkl file)
    :param num_instances: int
    :param output_dir: str (path to dir)
    :param export: bool
    :param export_file_path: str (path to .hdf5 file)
    :param load_params: bool
    :param save_params: bool
    :param label: str
    :param plot: bool
    :param interactive: bool
    :param debug: bool
    :param disp: bool
    :param framework: str
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)

    context.interface = get_parallel_interface(framework=framework, **kwargs)
    context.interface.start(disp=disp)
    context.interface.ensure_controller()
    nested_parallel_init_contexts_interactive(context, config_file_path=config_file_path,
                                              network_config_file_path=network_config_file_path,
                                              data_file_name_base=data_file_name_base, output_dir=output_dir,
                                              save_params=save_params, load_params=load_params, debug=debug,
                                              export_file_path=export_file_path, label=label, disp=disp, plot=plot,
                                              **kwargs)

    network_seeds, data_seeds = context.interface.execute(get_random_seeds)
    sequences = [network_seeds, data_seeds]

    context.interface.map(simulate, *sequences)

    if plot:
        context.interface.show()
        plt.show()

    if context.interactive:
        context.update(locals())
    else:
        context.interface.stop()


if __name__ == '__main__':
    main(standalone_mode=False)
