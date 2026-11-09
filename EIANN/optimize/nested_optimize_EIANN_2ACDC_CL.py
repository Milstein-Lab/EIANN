import os
import gc
import numpy as np

from EIANN import Network
from EIANN.utils import read_from_yaml, write_to_yaml
from EIANN.reinforcement_learning.environments import TaskSchedule, ScheduledEnv, make_2ACDC_tracks
from EIANN.reinforcement_learning.treadmill_metrics import block_rewards, seen_track_rewards
from nested.utils import Context, str_to_bool
from nested.optimize_utils import update_source_contexts
from EIANN.optimize.network_rl_config_updates import *
import EIANN.reinforcement_learning as reinforcement_learning
import EIANN.utils as utils


context = Context()

# 2ACDC_CL: continual (sequential) version of the 2ACDC task (Sun et al. 2025, Nature,
# https://www.nature.com/articles/s41586-024-08548-w). Instead of a random track every episode, the tracks are shown in
# blocks, as AlternatingSchedule orders them: episodes_per_task episodes of track 0, then of track 1, ..., repeated
# num_cycles times. Each block is one algo.learn() call followed by algo.end_task(), which calls
# network.update_CL_states() for CL learning rules (e.g. Backprop_SI) and first estimates offline importance for rules
# with importance_source='external' (e.g. Backprop_EWC) from the block's training episodes. val_interval applies within
# each block. Within a block the cue is redundant (licking at the current zone is enough), so the reward on the other track
# measures forgetting and interference. Epsilon decays over the whole run; the agent is not told when the task changes.
# Objectives, as in nested_optimize_EIANN_1_hidden_CL_mnist with a cumulative val set: for every block b,
# phase{b}_reward is the greedy reward averaged over the tracks seen so far, over the last num_episodes_reward_window
# episodes of the block. phase{b}_reward_track{j} (the full block x track reward matrix) and reward (the mean of the
# phase rewards) are features only.

# train and export a single seed (must be run from the EIANN package directory):
# python -m nested.analyze --framework=serial \
#   --config-file-path=optimize/optimize_config/2ACDC_CL/20261103_nested_optimize_EIANN_2_hidden_2ACDC_CL_van_bp_relu_SGD_config.yaml \
#   --param-file-path=optimize/optimize_params/2ACDC_CL/20261103_2ACDC_CL_params.yaml --model-key=van_bp_relu_SGD \
#   --output-dir=data/2ACDC_CL --label=van_bp_relu_SGD --num_instances=1 --export --disp
# for analysis with more blocks, add e.g. --num_cycles=4 and use a different --label (exports are not overwritten)

# added param: episodes_per_task (int) consecutive episodes on each track in a block
# added param: num_cycles (int) passes through all tracks; train_episodes = episodes_per_task * num_tracks * num_cycles
# added param: num_episodes_reward_window (int) last episodes of each block that phase rewards are averaged over
# added param: algorithm (str) name of an RLAlgorithm in EIANN.reinforcement_learning, e.g. QLearning or OnlineQLearning
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
    context.episodes_per_task = int(context.episodes_per_task)
    context.num_cycles = int(context.num_cycles)
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

    context.tracks = make_2ACDC_tracks(**context.treadmill_kwargs)
    context.num_tracks = len(context.tracks)
    context.num_blocks = context.num_tracks * context.num_cycles
    context.train_episodes = context.episodes_per_task * context.num_blocks


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

    if context.retrain:
        network = Network(context.layer_config, context.projection_config, seed=seed, **context.training_kwargs)
        algo = context.algorithm_class(network, persistent_state=network.persistent_state, seed=data_seed,
                                       **context.algorithm_kwargs)
        # one learn() per block, as MNIST CL trains one phase at a time; end_task() tells CL learning rules (SI, EWC)
        # that the task switched (exploration is not told). Each block's env trains on one track and validates on all
        for block in range(context.num_blocks):
            env = ScheduledEnv(TaskSchedule(context.tracks, order=[block % context.num_tracks]))
            algo.learn(env, context.episodes_per_task, status_bar=context.status_bar,
                       val_interval=context.val_interval)
            algo.end_task()
        # kept on the network so that it is exported with it
        network.rl_history = {'val_rewards': algo.val_rewards, 'val_episodes': algo.val_episodes,
                              'val_laps': algo.val_laps, 'episode_returns': algo.episode_returns,
                              'episode_lengths': algo.episode_lengths, 'loss_history': algo.loss_history,
                              'treadmill_kwargs': context.treadmill_kwargs,
                              'episodes_per_task': context.episodes_per_task, 'num_cycles': context.num_cycles,
                              'num_tracks': context.num_tracks}
    else:
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError('nested_optimize_EIANN_2ACDC_CL: retrain=False requires a network exported by a '
                                    'previous --export run; not found: %s' % data_file_path)
        network = utils.load_network(data_file_path, disp=context.disp)

    # rewards are sums of environment rewards, so only the loss shows that the weights diverged
    if not np.isfinite(network.rl_history['loss_history'][-1]):
        if context.interactive:
            context.update(locals())
        return dict()

    # entry [b, j]: greedy reward on track j at the end of block b; block b trains on track b % num_tracks
    rl_history = network.rl_history
    track_rewards = [[lap['episode_return'] for lap in laps] for laps in rl_history['val_laps']]
    num_blocks = rl_history['num_tracks'] * rl_history['num_cycles']
    reward_matrix = block_rewards(track_rewards, rl_history['val_episodes'], rl_history['episodes_per_task'],
                                  num_blocks, context.num_episodes_reward_window)
    # like the cumulative val set of MNIST CL: the mean over the tracks seen so far (0..block)
    phase_rewards = seen_track_rewards(reward_matrix)
    results = {}
    for block in range(num_blocks):
        results[f'phase{block}_reward'] = phase_rewards[block]
        for track in range(rl_history['num_tracks']):
            results[f'phase{block}_reward_track{track}'] = reward_matrix[block, track]
    # summary only (not an objective)
    results['reward'] = np.mean([results[f'phase{block}_reward'] for block in range(num_blocks)])

    if plot:
        from EIANN.reinforcement_learning.treadmill_plots import plot_2ACDC_summary  # keeps seaborn off cluster workers
        label = os.path.splitext(os.path.basename(data_file_path))[0]
        save_prefix = None if context.plot_dir is None else f"{context.plot_dir}/{label}"
        plot_2ACDC_summary(rl_history, title=f'{seed}_{data_seed}', save_prefix=save_prefix,
                           plot_format=context.plot_format)

    if export:
        config_dict = {'layer_config': context.layer_config,
                       'projection_config': context.projection_config,
                       'training_kwargs': context.training_kwargs,
                       'algorithm': context.algorithm,
                       'algorithm_kwargs': context.algorithm_kwargs}
        write_to_yaml(context.export_network_config_file_path, config_dict, convert_scalars=True)
        utils.save_network(network, path=data_file_path, disp=False)
        if context.disp:
            print('nested_optimize_EIANN_2ACDC_CL: pid: %i exported network config to %s and network history to %s' %
                  (os.getpid(), context.export_network_config_file_path, data_file_path))

    if not context.interactive:
        del network
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
    # one objective per block, like the per-phase accuracies of MNIST CL; per-track rewards and reward are features only
    max_reward = context.treadmill_kwargs['reward_values'][1]
    objectives = {f'phase{block}_reward': max_reward - features[f'phase{block}_reward']
                  for block in range(context.num_blocks)}
    return features, objectives
