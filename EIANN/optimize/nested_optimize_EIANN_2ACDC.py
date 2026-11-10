import os
import gc
import numpy as np

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml
from EIANN.reinforcement_learning.environments import RandomSchedule, ScheduledEnv, make_2ACDC_tracks
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_rl_config_updates import *
import EIANN.reinforcement_learning as reinforcement_learning
import EIANN.utils as utils


context = Context()

# 2ACDC task, after Sun et al. 2025, Nature (https://www.nature.com/articles/s41586-024-08548-w): two tracks with
# different cues and the same two reward zones, with the cue indicating which zone pays. Tracks are drawn at random
# every episode.

# train and export a single seed (must be run from the EIANN package directory):
# python -m nested.analyze --framework=serial \
#   --config-file-path=optimize/optimize_config/2ACDC/20261102_nested_optimize_EIANN_2_hidden_2ACDC_van_bp_relu_SGD_config.yaml \
#   --param-file-path=optimize/optimize_params/2ACDC/20261102_2ACDC_params.yaml --model-key=van_bp_relu_SGD \
#   --output-dir=data --label=van_bp_relu_SGD --num_instances=1 --export --disp

# added param: algorithm (str) name of an RLAlgorithm in EIANN.reinforcement_learning, e.g. QLearning or OnlineQLearning
# added param: qnext_network_config_file_path (str) network yaml of DPPQL's QNext network; when given, a QNext network is
#   built with its own seed and passed to the algorithm as qnext_network, and exported next to the Q network as
#   <name>_qnext.pkl and <export config>_qnext.yaml (default to None)
# added param: qnext_input_population (str) full name of the Q network population that QNext reads, passed to the
#   algorithm (default to the algorithm's, H1E)
# added param: algorithm_kwargs (dict) fixed arguments of the algorithm; update_context functions add the optimized ones
#   (e.g. update_q_learning_kwargs)
# added param: retrain (bool) with False, loads the network exported by a previous --export run instead of training it
#   (default to true)
# added param: plot_dir (str) with --plot, saves the figures to this folder instead of showing them (default to None)
# added param: plot_format (str) file format of the saved figures (default to svg)


def config_worker():
    context.seed_start = int(context.seed_start)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)
    context.data_seed_start = int(context.data_seed_start)
    context.status_bar = str_to_bool(context.status_bar)
    context.train_episodes = int(context.train_episodes)
    context.num_episodes_reward_window = int(context.num_episodes_reward_window)
    context.algorithm_class = getattr(reinforcement_learning, context.algorithm)

    if 'interactive' not in context():
        context.interactive = False
    else:
        context.interactive = str_to_bool(context.interactive)
    if 'retrain' not in context():
        context.retrain = True
    else:
        context.retrain = str_to_bool(context.retrain)
    if 'plot_dir' not in context():
        context.plot_dir = None
    if 'plot_format' not in context():
        context.plot_format = 'svg'

    context.network_name = context.network_config_file_path.split('/')[-1].split('.')[0]
    if 'export_network_config_file_path' not in context():
        if context.label is None:
            context.export_network_config_file_path = f"{context.output_dir}/{context.network_name}_optimized.yaml"
        else:
            context.export_network_config_file_path = \
                f"{context.output_dir}/{context.network_name}_{context.label}_optimized.yaml"

    network_config = read_from_yaml(context.network_config_file_path)
    context.layer_config = network_config['layer_config']
    context.projection_config = network_config['projection_config']
    context.training_kwargs = network_config['training_kwargs']

    if 'qnext_network_config_file_path' in context():
        qnext_network_config = read_from_yaml(context.qnext_network_config_file_path)
        context.qnext_layer_config = qnext_network_config['layer_config']
        context.qnext_projection_config = qnext_network_config['projection_config']
        context.qnext_training_kwargs = qnext_network_config['training_kwargs']
        if 'qnext_input_population' in context():
            context.algorithm_kwargs['qnext_input_population'] = context.qnext_input_population
    else:
        context.qnext_layer_config = None

    context.tracks = make_2ACDC_tracks(**context.treadmill_kwargs)


def get_random_seeds():
    network_seeds = [int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
                     for instance_id in range(context.seed_start, context.seed_start + context.num_instances)]
    data_seeds = [int.from_bytes((context.network_id, instance_id), byteorder='big')
                  for instance_id in range(context.data_seed_start, context.data_seed_start + context.num_instances)]
    return [network_seeds, data_seeds]


def get_data_file_path(seed, data_seed):
    """
    Path of the pickled network of one instance. Used both to export it and to load it back with retrain=False.

    :param seed: int
    :param data_seed: int
    :return: str
    """
    label = '' if context.label is None else f"_{context.label}"
    return f"{context.output_dir}/{context.network_name}/{context.network_name}_{seed}_{data_seed}{label}.pkl"


def get_qnext_path(path):
    """
    Path of the QNext network (DPPQL) or its config, saved next to the Q network's: <name>_qnext.<ext>.
    """
    root, ext = os.path.splitext(path)
    return f"{root}_qnext{ext}"


def compute_features(x, seed, data_seed, model_id=None, export=False, plot=False):
    """

    :param x: array of float
    :param seed: int
    :param data_seed: int
    :param model_id: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    update_source_contexts(x, context)
    data_file_path = get_data_file_path(seed, data_seed)

    qnext_network = None
    if context.retrain:
        network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)
        algorithm_kwargs = dict(context.algorithm_kwargs)
        if context.qnext_layer_config is not None:
            # DPPQL: network seeds are 3-byte ints, so this offset gives QNext a seed distinct from every Q network's
            qnext_network = Network(context.qnext_layer_config, context.qnext_projection_config, seed=seed + 2 ** 24,
                                    **context.qnext_training_kwargs)
            algorithm_kwargs['qnext_network'] = qnext_network
        algo = context.algorithm_class(network, persistent_state=network.persistent_state, seed=data_seed,
                                       **algorithm_kwargs)
        env = ScheduledEnv(RandomSchedule(context.tracks, seed=data_seed))
        algo.learn(env, context.train_episodes, status_bar=context.status_bar, val_interval=context.val_interval)
        # kept on the network so that it is exported with it
        network.rl_history = {'val_rewards': algo.val_rewards, 'val_episodes': algo.val_episodes,
                              'val_laps': algo.val_laps, 'episode_returns': algo.episode_returns,
                              'episode_lengths': algo.episode_lengths, 'loss_history': algo.loss_history,
                              'treadmill_kwargs': context.treadmill_kwargs}
        if qnext_network is not None:
            network.rl_history['qnext_loss_history'] = algo.qnext_loss_history
    else:
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError('nested_optimize_EIANN_2ACDC: retrain=False requires a network exported by a '
                                    'previous --export run; not found: %s' % data_file_path)
        network = utils.load_network(data_file_path, disp=context.disp)
        if context.qnext_layer_config is not None:
            qnext_network = utils.load_network(get_qnext_path(data_file_path), disp=context.disp)

    # rewards are sums of environment rewards, so only the loss shows that the weights diverged
    if not np.isfinite(network.rl_history['loss_history'][-1]):
        if context.interactive:
            context.update(locals())
        return dict()

    num_val_steps_reward_window = context.num_episodes_reward_window // int(context.val_interval[2])
    # relative to a lap without licks (every lap has the same length), so rewards shifted to be non-negative, e.g.
    # (0.1, 1.1, 0.) instead of (0., 1., -0.1), give the same feature as the unshifted task
    results = {'reward': np.mean(network.rl_history['val_rewards'][-num_val_steps_reward_window:]) -
                         context.treadmill_kwargs['length'] * context.treadmill_kwargs['reward_values'][0]}

    if plot:
        from EIANN.reinforcement_learning.treadmill_plots import plot_2ACDC_summary  # keeps seaborn off cluster workers
        label = os.path.splitext(os.path.basename(data_file_path))[0]
        save_prefix = None if context.plot_dir is None else f"{context.plot_dir}/{label}"
        plot_2ACDC_summary(network.rl_history, title=f'{seed}_{data_seed}', save_prefix=save_prefix,
                           plot_format=context.plot_format)

    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs,
                       'algorithm': context.algorithm,
                       'algorithm_kwargs': context.algorithm_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        utils.save_network(network, path=data_file_path, disp=False)
        if qnext_network is not None:
            # saved separately: an nn.Module attribute of network would become one of its submodules
            qnext_config_dict = {'layer_config': context.qnext_layer_config,
                                 'projection_config': context.qnext_projection_config,
                                 'training_kwargs': context.qnext_training_kwargs}
            write_to_yaml(get_qnext_path(context.export_network_config_file_path), qnext_config_dict,
                          convert_scalars=True)
            utils.save_network(qnext_network, path=get_qnext_path(data_file_path), disp=False)
        if context.disp:
            print('nested_optimize_EIANN_2ACDC: pid: %i exported network config to %s and network history to %s' %
                  (os.getpid(), context.export_network_config_file_path, data_file_path))

    if not context.interactive:
        del network
        del qnext_network
        gc.collect()
    else:
        context.update(locals())

    return results


def filter_features(primitives, current_features, model_id=None, export=False, plot=False):

    features = {}
    for instance_features in primitives:
        for key, val in instance_features.items():
            if np.isnan(val) or np.isinf(val):
                return dict()
            if key not in features:
                features[key] = []
            features[key].append(val)
    for key, val in features.items():
        features[key] = np.mean(val)

    return features


def get_objectives(features, model_id=None, export=False, plot=False):
    # one correct lick, relative to a lap without licks (see compute_features)
    max_reward = context.treadmill_kwargs['reward_values'][1] - context.treadmill_kwargs['reward_values'][0]
    objectives = {'reward': max_reward - features['reward']}
    return features, objectives
