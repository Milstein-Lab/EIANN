import itertools

import numpy as np
import gymnasium
from gymnasium import spaces


class CueTreadmillEnv(gymnasium.Env):
    """
    1D treadmill with a cue and several reward zones. The agent advances one cell per step whatever it does, and at
    each cell either ignores (action 0) or licks (action 1). Licking at the last cell of the rewarded zone is correct;
    every other lick is penalized; ignoring is neutral. One episode is one lap.

    The observation is a one-hot encoding of the current cell: [plain, cue_0 .. cue_{num_cues-1}, zone_0 .. zone_{n-1}].
    Every reward zone is marked on every track, so the cue is needed to know which zone pays. Each instance is one fixed
    track (cue and rewarded zone); a TaskSchedule chooses between tracks.
    """
    metadata = {'render_modes': ['ansi']}

    def __init__(self, length=23, cue_position=8, cue_length=3, num_cues=2, cue=0, reward_positions=(15, 19),
                 reward_length=2, rewarded_zone=0, reward_values=(0., 1., -0.1), render_mode=None):
        """
        Parameters
        ----------
        length : int, default 23
            Number of cells (steps) in one lap.
        cue_position : int, default 8
            First cell of the cue.
        cue_length : int, default 3
        num_cues : int, default 2
            Number of distinct cues across tracks (sets the observation size).
        cue : int, default 0
            Cue shown on this track.
        reward_positions : tuple of int, default (15, 19)
            Last cell of each reward zone.
        reward_length : int, default 2
            Number of cells in each reward zone.
        rewarded_zone : int, default 0
            Index into reward_positions of the zone that pays on this track.
        reward_values : tuple of float, default (0., 1., -0.1)
            Reward for (ignoring, licking correctly, licking incorrectly).
        render_mode : str, optional
            'ansi' or None.
        """
        reward_positions = tuple(sorted(reward_positions))
        assert 0 <= cue < num_cues
        assert 0 <= rewarded_zone < len(reward_positions)
        assert cue_position >= 0 and cue_position + cue_length <= reward_positions[0] - reward_length + 1, \
            'CueTreadmillEnv: the cue must end before the first reward zone'
        assert all(later - earlier >= reward_length for earlier, later in zip(reward_positions, reward_positions[1:])), \
            'CueTreadmillEnv: reward zones must not overlap'
        assert reward_positions[-1] < length
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.length = length
        self.cue_position = cue_position
        self.cue_length = cue_length
        self.num_cues = num_cues
        self.cue = cue
        self.reward_positions = reward_positions
        self.reward_length = reward_length
        self.rewarded_zone = rewarded_zone
        self.rewarded_position = reward_positions[rewarded_zone]
        self.reward_values = reward_values
        self.render_mode = render_mode

        obs_size = 1 + num_cues + len(reward_positions)
        self.observation_space = spaces.Box(0., 1., (obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.cell_observations = np.zeros((length, obs_size), dtype=np.float32)
        self.cell_observations[:, 0] = 1.
        self.cell_observations[cue_position:cue_position + cue_length, 0] = 0.
        self.cell_observations[cue_position:cue_position + cue_length, 1 + cue] = 1.
        for zone, position in enumerate(reward_positions):
            self.cell_observations[position - reward_length + 1:position + 1, 0] = 0.
            self.cell_observations[position - reward_length + 1:position + 1, 1 + num_cues + zone] = 1.
        self.terminal_observation = np.zeros(obs_size, dtype=np.float32)

        self.position = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return self.cell_observations[self.position].copy(), {}

    def step(self, action):
        if action == 1:
            correct = self.position == self.rewarded_position
            reward = self.reward_values[1] if correct else self.reward_values[2]
        else:
            reward = self.reward_values[0]

        self.position += 1
        terminated = self.position == self.length
        obs = self.terminal_observation.copy() if terminated else self.cell_observations[self.position].copy()
        return obs, float(reward), terminated, False, {}

    def get_zone_spans(self):
        """
        Returns
        -------
        cue_span : tuple of int
            (start, stop) cells of the cue, with stop exclusive.
        reward_spans : list of tuple of int
            (start, stop) cells of each reward zone.
        """
        cue_span = (self.cue_position, self.cue_position + self.cue_length)
        return cue_span, [(position - self.reward_length + 1, position + 1) for position in self.reward_positions]

    def get_regions(self):
        """
        Regions of the track for comparing representations across tracks. Needs two reward zones.

        Returns
        -------
        dict of tuple of int
            (start, stop) cells of the region before the cue, the cue, the cell before each reward zone, and the
            region from the last rewarded cell to the end.
        """
        cue_span, reward_spans = self.get_zone_spans()
        return {'initial_region': (0, cue_span[0]),
                'indicator': cue_span,
                'pre_r1': (reward_spans[0][0] - 1, reward_spans[0][0]),
                'pre_r2': (reward_spans[1][0] - 1, reward_spans[1][0]),
                'end_region': (self.reward_positions[1], self.length)}

    def get_off_diagonal_mask(self):
        """
        Pairs of cells to compare two tracks away from the diagonal: every pair from two different plain stretches of
        the track, which are the stretches separated by the cue and the two reward zones.

        Returns
        -------
        np.ndarray of bool
            Shape (length, length).
        """
        cue_span, reward_spans = self.get_zone_spans()
        plain_stretches = [(0, cue_span[0]), (cue_span[1], reward_spans[0][0]), (reward_spans[0][1], reward_spans[1][0]),
                           (reward_spans[1][1], self.length)]
        mask = np.zeros((self.length, self.length), dtype=bool)
        for (row_start, row_stop), (col_start, col_stop) in itertools.permutations(plain_stretches, 2):
            mask[row_start:row_stop, col_start:col_stop] = True
        return mask

    def render(self):
        if self.render_mode != 'ansi':
            return None
        cells = []
        for position, cell_obs in enumerate(self.cell_observations):
            index = int(np.argmax(cell_obs))
            if index == 0:
                cell = '.'
            elif index <= self.num_cues:
                cell = f'c{index - 1}'
            else:
                cell = f'z{index - 1 - self.num_cues}'
            if position == self.rewarded_position:
                cell += '*'
            if position == self.position:
                cell = f'[{cell}]'
            cells.append(cell)
        return ' '.join(cells)


def make_2ACDC_tracks(reward_positions=(15, 19), **kwargs):
    """
    The tracks of the 2ACDC task (Sun et al. 2025, Nature): one per reward zone, with cue i indicating that zone i pays.

    Parameters
    ----------
    reward_positions : tuple of int, default (15, 19)
    **kwargs
        Passed to CueTreadmillEnv.

    Returns
    -------
    list of CueTreadmillEnv
    """
    return [CueTreadmillEnv(cue=zone, rewarded_zone=zone, reward_positions=reward_positions, **kwargs)
            for zone in range(len(reward_positions))]
