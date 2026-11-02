import gymnasium


class ScheduledEnv(gymnasium.Env):
    """
    Environment that switches to the schedule's next env on every reset, e.g. a random track every episode with
    ScheduledEnv(RandomSchedule(tracks)). A finite schedule is restarted when it runs out. info['task'] holds the index of the
    current env.
    """
    def __init__(self, schedule):
        """
        Parameters
        ----------
        schedule : TaskSchedule
            All of its envs must share observation and action spaces.
        """
        self.schedule = schedule
        self.envs = schedule.envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        assert all(env.observation_space == self.observation_space and env.action_space == self.action_space
                   for env in self.envs), 'ScheduledEnv: all envs must share observation and action spaces'
        self._task_iter = iter(schedule)
        self.task = None
        self.env = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.task = next(self._task_iter)
        except StopIteration:
            self._task_iter = iter(self.schedule)
            self.task = next(self._task_iter)
        self.env = self.envs[self.task]
        obs, info = self.env.reset(seed=seed, options=options)
        info['task'] = self.task
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['task'] = self.task
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
