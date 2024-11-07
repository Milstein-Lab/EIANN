

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
