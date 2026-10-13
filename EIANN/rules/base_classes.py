import torch


class ContinualLearningMixin(object):
    """
    Task bookkeeping shared by continual-learning (CL) learning rules. network.update_CL_states() calls
    update_CL_states() on every projection's learning rule between tasks. When task_incremental is True, learning only
    sees the output units of the current task's classes (task_classes[task_num]).
    Rules using this mixin call init_continual_learning() in __init__, and then either:
      - backprop rules: restrict the loss to the current task's output units (e.g. Backprop_CL.task_loss), or
      - local rules driven by the output error (target - output activity): replace the target with task_target(), so
        the output error is exactly zero on units outside the current task.
    """
    
    def init_continual_learning(self, task_incremental=False, task_classes=None):
        self.task_incremental = bool(task_incremental)
        if self.task_incremental and task_classes is None:
            raise ValueError('%s: task_classes must be provided when task_incremental is True' %
                             self.__class__.__name__)
        self.task_classes = task_classes
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
    
    def get_task_classes(self, task_num=None):
        if task_num is None:
            task_num = self.task_num
        return list(self.task_classes[task_num])
    
    def task_target(self, network, target, task_num=None):
        """
        Target with every output unit outside the current task set to that unit's current activity, so that
        (target - output activity) is zero outside the current task.
        :param network: :class:'Network'
        :param target: tensor
        :param task_num: int; defaults to the current task
        :return: tensor
        """
        if not self.task_incremental:
            return target
        task_target = network.output_pop.activity.detach().clone()
        task_idx = self.get_task_classes(task_num)
        task_target[..., task_idx] = target[..., task_idx]
        return task_target


class LearningRule(object):
    def __init__(self, projection, learning_rate=None):
        self.projection = projection
        if learning_rate is None:
            learning_rate = self.projection.post.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    def reinit(self):
        pass

    def update(self):
        pass

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        pass
    
    @classmethod
    def shared_backward_methods(cls, learning_rule):
        return learning_rule.__class__.backward.__func__ is cls.backward.__func__


class BiasLearningRule(object):
    def __init__(self, population, learning_rate=None):
        self.population = population
        if learning_rate is None:
            learning_rate = self.population.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    def reinit(self):
        pass

    def update(self):
        pass

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        pass
    
    @classmethod
    def shared_backward_methods(cls, learning_rule):
        return learning_rule.__class__.backward.__func__ is cls.backward.__func__
