import gymnasium
import minigrid  # registers the MiniGrid-* environment ids with gymnasium
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation

from .task_schedules import TaskSchedule, SequentialSchedule, AlternatingSchedule, RandomSchedule
from .scheduled_env import ScheduledEnv
from .cue_treadmill import CueTreadmillEnv, make_2ACDC_tracks
from .crossing import CrossingChoiceEnv, make_crossing_tasks, CROSSING_TASKS

gymnasium.register('EIANN/CueTreadmill-v0',
                   entry_point='EIANN.reinforcement_learning.environments.cue_treadmill:CueTreadmillEnv')
gymnasium.register('EIANN/Crossing-v0',
                   entry_point='EIANN.reinforcement_learning.environments.crossing:CrossingChoiceEnv')


def make_env(env_id, seed=None, **kwargs):
    """
    Build a gymnasium environment whose observations are flat vectors that can be passed to an EIANN Network.

    Parameters
    ----------
    env_id : str
        Registered gymnasium id (e.g. 'MiniGrid-Empty-5x5-v0').
    seed : int, optional
        Seeds the environment's random number generator and action space.
    **kwargs
        Passed to gymnasium.make.

    Returns
    -------
    gymnasium.Env
    """
    env = gymnasium.make(env_id, **kwargs)
    if env_id.startswith('MiniGrid'):
        # keep only the partially observable (7, 7, 3) image and drop the direction and mission string
        env = FlattenObservation(ImgObsWrapper(env))
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
