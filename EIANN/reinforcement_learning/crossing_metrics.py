"""
Metrics of the Crossing_CL task (CrossingChoiceEnv), from the greedy episodes recorded during training
(RLAlgorithm.val_laps). Every episode ends in one outcome: 'correct', 'wrong' (another goal tile) or 'timeout'.
The block x task matrices are averaged over the same window as the 2ACDC_CL rewards (treadmill_metrics.block_rewards),
so ACC/BWT/FWT can be computed from them the same way as for MNIST CL.
"""
import numpy as np

from .treadmill_metrics import block_rewards


def stack_outcomes(val_laps):
    """
    Parameters
    ----------
    val_laps : list of list of dict
        run_greedy_episode records, by validation step and task.

    Returns
    -------
    np.ndarray of str
        Shape (num_val, num_tasks).
    """
    outcomes = np.array([[lap['outcome'] for lap in laps] for laps in val_laps], dtype=object)
    if np.any(outcomes == None):
        raise ValueError('stack_outcomes: a greedy episode has no outcome (not recorded from a CrossingChoiceEnv?)')
    return outcomes.astype(str)


def block_outcome_rates(val_laps, episodes, episodes_per_task, num_blocks, window):
    """
    Outcome rates on every task at the end of every block, averaged over the validation steps in the last window
    episodes of the block.

    Parameters
    ----------
    val_laps : list of list of dict
    episodes : np.ndarray of int
        Episode of each validation step (RLAlgorithm.val_episodes).
    episodes_per_task : int
    num_blocks : int
    window : int

    Returns
    -------
    dict of np.ndarray
        Each of shape (num_blocks, num_tasks):
        success: fraction of episodes that reached the rewarded tile;
        navigation: fraction that reached any goal tile (did not time out);
        choice: success / navigation, the fraction of choices that were correct (NaN where no tile was reached).
    """
    outcomes = stack_outcomes(val_laps)
    success = block_rewards(outcomes == 'correct', episodes, episodes_per_task, num_blocks, window)
    navigation = block_rewards(outcomes != 'timeout', episodes, episodes_per_task, num_blocks, window)
    with np.errstate(invalid='ignore', divide='ignore'):
        choice = np.where(navigation > 0, success / navigation, np.nan)
    return {'success': success, 'navigation': navigation, 'choice': choice}


def episode_positions(env, actions):
    """
    Replay a recorded episode (e.g. the actions of a run_greedy_episode record) on a CrossingChoiceEnv, which is
    deterministic, to recover where the agent went.

    Parameters
    ----------
    env : CrossingChoiceEnv
        The task the episode was run on.
    actions : sequence of int

    Returns
    -------
    positions : np.ndarray of int
        Shape (len(actions) + 1, 2): (row, col) at the start and after every action.
    outcome : str or None
        info['outcome'] of the last step ('correct', 'wrong' or 'timeout'), or None if the episode did not end.
    """
    env.reset()
    positions = [env.start_pos]
    outcome = None
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(int(action))
        x, y = env.agent_pos
        positions.append((y - 1, x - 1))
        outcome = info.get('outcome', outcome)
    return np.array(positions, dtype=int), outcome


def cl_summary(matrix, baseline=None):
    """
    Continual-learning summary of a block x task matrix where block i trains on task i (one cycle), as for MNIST CL
    (Diaz-Rodriguez et al. 2018, arXiv:1810.13166).

    Parameters
    ----------
    matrix : np.ndarray
        Shape (num_tasks, num_tasks), e.g. block_outcome_rates(...)['success'] for num_cycles=1. Entry [i, j]: task j
        after training on tasks 0..i.
    baseline : np.ndarray, optional
        Shape (num_tasks,): each task before any training (e.g. the greedy success at the first validation step), for FWT.

    Returns
    -------
    dict of float
        ACC: mean over tasks after the last block;
        BWT: mean change on every earlier task between the end of its own block and the end of the last block;
        FWT: mean of task j after block j - 1 minus its baseline (NaN without a baseline).
    """
    matrix = np.asarray(matrix, dtype=float)
    num_tasks = matrix.shape[1]
    assert matrix.shape == (num_tasks, num_tasks), 'cl_summary: needs one block per task (num_cycles=1)'
    acc = np.mean(matrix[-1])
    bwt = np.mean([matrix[-1, task] - matrix[task, task] for task in range(num_tasks - 1)])
    fwt = np.nan if baseline is None else \
        np.mean([matrix[task - 1, task] - baseline[task] for task in range(1, num_tasks)])
    return {'ACC': float(acc), 'BWT': float(bwt), 'FWT': float(fwt)}
