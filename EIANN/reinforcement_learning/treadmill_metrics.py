import warnings

import numpy as np


def stack_laps(val_laps, val_episodes):
    """
    Stack the greedy laps recorded during training (RLAlgorithm.val_laps) into arrays. All of the metrics and plots of
    treadmill tasks work from this.

    Parameters
    ----------
    val_laps : list of list of dict
        run_greedy_episode records, by validation step and track.
    val_episodes : list of int
        Episode of each validation step.

    Returns
    -------
    dict
        episodes: shape (num_val,)
        rewards: shape (num_val, num_tracks)
        q_values: shape (num_val, num_tracks, length, num_actions)
        actions: shape (num_val, num_tracks, length)
        activity: dict of population name to shape (num_val, num_tracks, length, population size)
    """
    def stack(get_value):
        return np.array([[get_value(lap) for lap in laps] for laps in val_laps])

    return {'episodes': np.array(val_episodes),
            'rewards': stack(lambda lap: lap['episode_return']),
            'q_values': stack(lambda lap: lap['q_values']),
            'actions': stack(lambda lap: lap['actions']),
            'activity': {name: stack(lambda lap: lap['activity'][name]) for name in val_laps[0][0]['activity']}}


def nanmean(values):
    with warnings.catch_warnings():
        # a block of only NaNs (e.g. silent activity) has a NaN mean
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return float(np.nanmean(values))


def cross_correlation(activity, nonzero_only=True):
    """
    Cross-correlation of activity across tracks. For each pair of tracks (i, j), entry [a, b] is the Pearson correlation
    across units between the activity at position a of track i and at position b of track j.

    Parameters
    ----------
    activity : np.ndarray
        Shape (num_tracks, length, num_units).
    nonzero_only : bool, default True
        Whether to drop units that are silent at all positions of both tracks, so they do not inflate the correlation.

    Returns
    -------
    np.ndarray
        Shape (num_track_pairs, length, length). Positions with constant activity give NaN.
    """
    matrices = []
    for i in range(len(activity)):
        for j in range(i + 1, len(activity)):
            track_i, track_j = np.nan_to_num(activity[i]), np.nan_to_num(activity[j])
            if nonzero_only:
                active = (np.abs(track_i).sum(axis=0) > 0) | (np.abs(track_j).sum(axis=0) > 0)
                track_i, track_j = track_i[:, active], track_j[:, active]
            length = track_i.shape[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                matrices.append(np.corrcoef(track_i, track_j)[:length, length:])
    return np.array(matrices)


def region_cross_correlation(matrix, track):
    """
    Mean of a cross-correlation matrix within the regions of a track.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (length, length).
    track : CueTreadmillEnv
        Defines the regions.

    Returns
    -------
    dict of float
        The mean within each region of track.get_regions() (same positions on both tracks), and 'off_diagonal', the
        mean over track.get_off_diagonal_mask().
    """
    means = {name: nanmean(matrix[start:stop, start:stop]) for name, (start, stop) in track.get_regions().items()}
    means['off_diagonal'] = nanmean(matrix[track.get_off_diagonal_mask()])
    return means


def cross_correlation_history(laps, population):
    """
    Returns
    -------
    np.ndarray
        Cross-correlation of a population at every validation step, shape (num_val, num_track_pairs, length, length).
    """
    return np.array([cross_correlation(activity) for activity in laps['activity'][population]])


def region_cross_correlation_history(laps, population, track):
    """
    Returns
    -------
    dict of np.ndarray
        Region means of a population at every validation step, each of shape (num_val, num_track_pairs).
    """
    matrices = cross_correlation_history(laps, population)
    means = [[region_cross_correlation(matrix, track) for matrix in pairs] for pairs in matrices]
    return {name: np.array([[pair[name] for pair in pairs] for pairs in means]) for name in means[0][0]}


def licks_by_zone(laps, track):
    """
    Returns
    -------
    np.ndarray
        Number of licks inside each reward zone of each track at every validation step, shape (num_val, num_tracks,
        num_zones).
    """
    cue_span, reward_spans = track.get_zone_spans()
    return np.stack([laps['actions'][:, :, start:stop].sum(axis=2) for start, stop in reward_spans], axis=2)


def block_rewards(rewards, episodes, episodes_per_task, num_blocks, window):
    """
    Greedy reward on every track at the end of every block of a blocked schedule (AlternatingSchedule), like the
    accuracy matrix of continual learning: entry [b, j] is the mean reward on track j over the validation steps in the
    last window episodes of block b.

    Parameters
    ----------
    rewards : np.ndarray
        Shape (num_val, num_tracks), e.g. stack_laps(...)['rewards'].
    episodes : np.ndarray of int
        Episode of each validation step, shape (num_val,).
    episodes_per_task : int
        Length of a block, in episodes.
    num_blocks : int
    window : int
        Number of episodes at the end of each block to average over.

    Returns
    -------
    np.ndarray
        Shape (num_blocks, num_tracks). A block without validation steps in its window gives NaN.
    """
    rewards = np.asarray(rewards, dtype=float)
    episodes = np.asarray(episodes)
    matrix = np.full((num_blocks, rewards.shape[1]), np.nan)
    for block in range(num_blocks):
        stop = (block + 1) * episodes_per_task
        in_window = (episodes >= stop - window) & (episodes < stop)
        if np.any(in_window):
            matrix[block] = rewards[in_window].mean(axis=0)
    return matrix


def seen_track_rewards(reward_matrix):
    """
    Reward of every block averaged over the tracks seen so far, like the cumulative val set of class-incremental MNIST:
    block b of an AlternatingSchedule averages tracks 0..min(b, num_tracks - 1). These are the 2ACDC_CL objectives.

    Parameters
    ----------
    reward_matrix : np.ndarray
        Shape (num_blocks, num_tracks), from block_rewards.

    Returns
    -------
    np.ndarray
        Shape (num_blocks,).
    """
    num_blocks, num_tracks = reward_matrix.shape
    return np.array([np.mean(reward_matrix[block, :min(block + 1, num_tracks)]) for block in range(num_blocks)])
