import numpy as np


class TaskSchedule(object):
    """
    Order in which a fixed set of tasks (environments) is used. Iterating yields task indexes into envs; ScheduledEnv
    takes one index per episode.

    New schedules subclass TaskSchedule and implement __iter__ as a generator, or pass an explicit order.
    """
    def __init__(self, envs, order=None):
        """
        Parameters
        ----------
        envs : list of gymnasium.Env
        order : list of int, optional
            Task index of each phase.
        """
        self.envs = list(envs)
        self.order = order

    def __iter__(self):
        if self.order is None:
            raise NotImplementedError(f'{type(self).__name__}: pass an order or implement __iter__')
        yield from self.order


class SequentialSchedule(TaskSchedule):
    """
    Each task once, in order: 0, 1, ..., n-1.
    """
    def __iter__(self):
        yield from range(len(self.envs))


class AlternatingSchedule(TaskSchedule):
    """
    Cycle through all tasks num_cycles times, repeating each task episodes_per_task times: with episodes_per_task=2,
    0, 0, 1, 1, ..., n-1, n-1, 0, 0, ...
    """
    def __init__(self, envs, num_cycles=2, episodes_per_task=1):
        """
        Parameters
        ----------
        envs : list of gymnasium.Env
        num_cycles : int, default 2
        episodes_per_task : int, default 1
        """
        super().__init__(envs)
        self.num_cycles = num_cycles
        self.episodes_per_task = episodes_per_task

    def __iter__(self):
        for cycle in range(self.num_cycles):
            for task in range(len(self.envs)):
                for repeat in range(self.episodes_per_task):
                    yield task


class RandomSchedule(TaskSchedule):
    """
    Tasks drawn uniformly at random, num_phases times, or endlessly when num_phases is None.
    """
    def __init__(self, envs, num_phases=None, seed=None):
        super().__init__(envs)
        self.num_phases = num_phases
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        phase = 0
        while self.num_phases is None or phase < self.num_phases:
            yield int(self.rng.integers(len(self.envs)))
            phase += 1
