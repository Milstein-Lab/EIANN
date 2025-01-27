import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os, sys, math
from copy import deepcopy
import numpy as np
import h5py
import gc
import click
import matplotlib.pyplot as plt

from EIANN import Network
from EIANN.utils import (read_from_yaml, write_to_yaml, analyze_simple_EIANN_epoch_loss_and_accuracy, \
    sort_by_val_history, recompute_validation_loss_and_accuracy, check_equilibration_dynamics, \
    recompute_train_loss_and_accuracy, compute_test_loss_and_accuracy_history, sort_by_class_averaged_val_output,
                         get_binned_mean_population_attribute_history_dict)
from EIANN.plot import (plot_batch_accuracy, plot_train_loss_history, plot_validate_loss_history, plot_receptive_fields,
                        plot_representation_metrics)
from nested.utils import Context, get_unknown_click_arg_dict, str_to_bool
from nested.parallel import get_parallel_interface
from nested.optimize_utils import nested_parallel_init_contexts_interactive
import EIANN.optimize.nested_optimize_EIANN_1_hidden_mnist
from EIANN.optimize.nested_optimize_EIANN_1_hidden_mnist import get_random_seeds
import EIANN.utils as utils


context = Context()

context.test = 'test'


def config_worker():
    EIANN.optimize.nested_optimize_EIANN_1_hidden_mnist.context = context
    EIANN.optimize.nested_optimize_EIANN_1_hidden_mnist.config_worker()
    

def simulate(seed, data_seed, data_file_path=None, export=False, plot=False):
    """

    :param seed: int
    :param data_seed: int
    :param data_file_path: str (path)
    :param export: bool
    :param plot: bool
    """
    data_generator = context.data_generator
    train_sub_dataloader = context.train_sub_dataloader
    val_dataloader = context.val_dataloader
    test_dataloader = context.test_dataloader
    
    epochs = context.epochs
    
    network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)
    
    if plot:
        if context.plot_initial:
            title = 'Initial (%i, %i)' % (seed, data_seed)
            plot_batch_accuracy(network, test_dataloader, population='all', title=title)
    
    if data_file_path is None:
        network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
        if context.label is None:
            data_file_path = f"{context.output_dir}/{network_name}_{seed}_{data_seed}.pkl"
        else:
            data_file_path = f"{context.output_dir}/{network_name}_{seed}_{data_seed}_{context.label}.pkl"
    
    if os.path.exists(data_file_path) and not context.retrain:
        network = utils.load_network(data_file_path)
        if context.disp:
            print('nested_optimize_EIANN_1_hidden_mnist: pid: %i loaded network history from %s' %
                  (os.getpid(), data_file_path))
    else:
        data_generator.manual_seed(data_seed)
        if context.debug:
            import time
            current_time = time.time()
        network.train(train_sub_dataloader, val_dataloader, epochs=epochs,
                      val_interval=context.val_interval,  # e.g. (-201, -1, 10),
                      samples_per_epoch=context.train_steps, store_history=context.store_history,
                      store_dynamics=context.store_dynamics, store_history_interval=context.store_history_interval,
                      store_params=context.store_params, store_params_interval=context.store_params_interval,
                      status_bar=context.status_bar, debug=context.debug)
        
        # reorder output units if using unsupervised learning rule
        if not context.supervised:
            if context.eval_accuracy == 'final':
                min_loss_idx = len(network.val_loss_history) - 1
                sorted_output_idx = sort_by_class_averaged_val_output(network, val_dataloader)
            elif context.eval_accuracy == 'best':
                min_loss_idx, sorted_output_idx = sort_by_val_history(network, val_dataloader, plot=plot)
            else:
                raise Exception('nested_optimize_EIANN_1_hidden_mnist: eval_accuracy must be final or best, not %s' %
                                context.eval_accuracy)
            sorted_val_loss_history, sorted_val_accuracy_history = \
                recompute_validation_loss_and_accuracy(network, val_dataloader, sorted_output_idx=sorted_output_idx,
                                                       store=True)
        else:
            min_loss_idx = torch.argmin(network.val_loss_history)
            sorted_output_idx = None
            sorted_val_loss_history = network.val_loss_history
            sorted_val_accuracy_history = network.val_accuracy_history
        
        if context.store_history and (context.store_history_interval is None):
            binned_train_loss_steps, sorted_train_loss_history, sorted_train_accuracy_history = \
                recompute_train_loss_and_accuracy(network, sorted_output_idx=sorted_output_idx, plot=plot)
    
    if plot:
        title = 'Final (%i, %i)' % (seed, data_seed)
        plot_batch_accuracy(network, test_dataloader, population='all', sorted_output_idx=sorted_output_idx,
                            title=title)
        plot_train_loss_history(network)
        plot_validate_loss_history(network)
    
    if 'H1' in network.layers:
        if context.compute_receptive_fields:
            # Compute receptive fields
            population = network.H1.E
            receptive_fields = utils.compute_maxact_receptive_fields(population)
        else:
            receptive_fields = network.H1.E.Input.E.weight.detach()
        
        if plot:
            plot_receptive_fields(receptive_fields, sort=True, num_cols=10, num_rows=10)
    
    if context.full_analysis:
        if 'H1' in network.layers:
            metrics_dict = utils.compute_representation_metrics(network.H1.E, test_dataloader, receptive_fields)
            plot_representation_metrics(metrics_dict)
        test_loss_history, test_accuracy_history = \
            compute_test_loss_and_accuracy_history(network, test_dataloader, sorted_output_idx=sorted_output_idx,
                                                   plot=plot, status_bar=context.status_bar)
    
    if context.constrain_equilibration_dynamics or context.debug:
        residuals = check_equilibration_dynamics(network, test_dataloader, context.equilibration_activity_tolerance,
                                                 store_num_steps=context.store_num_steps, disp=context.disp, plot=plot)
        if residuals > 0. and context.disp:
            print('Failed equilibration dynamics test (%i, %i)' % (seed, data_seed))
            sys.stdout.flush()
    
    if export:
        utils.save_network(network, path=data_file_path, disp=False)
        if context.disp:
            print('simulate_EIANN_2_hidden_mnist: pid: %i exported network history to %s' %
                  (os.getpid(), data_file_path))
    
    if not context.interactive:
        del network
        gc.collect()
    else:
        context.update(locals())


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/mnist/simulate_EIANN_1_hidden_mnist_supervised_config.yaml')
@click.option("--network-config-file-path", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--data-file-path", '-d', multiple=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='../data')
@click.option("--export", is_flag=True)
@click.option("--retrain", type=bool, default=True)
@click.option("--label", type=str, default=None)
@click.option("--plot", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--disp", is_flag=True)
@click.option("--framework", type=str, default='serial')
@click.pass_context
def main(cli, config_file_path, network_config_file_path, data_file_path, output_dir, export, retrain, label, plot,
         interactive, debug, disp, framework):
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
    :param data_file_path: tuple of str (path to .pkl file)
    :param output_dir: str (path to dir)
    :param export: bool
    :param retrain: bool
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
                                              network_config_file_path=network_config_file_path, output_dir=output_dir,
                                              retrain=retrain, debug=debug, label=label, disp=disp, plot=plot, **kwargs)
    
    if data_file_path:
        context.num_instances = len(data_file_path)
        data_file_path_list = list(data_file_path)
    else:
        data_file_path_list = [None] * context.num_instances
    
    network_seeds, data_seeds = context.interface.execute(get_random_seeds)
    sequences = [network_seeds, data_seeds, data_file_path_list, [export] * context.num_instances,
                 [plot] * context.num_instances]

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
