from .base_classes import LearningRule, BiasLearningRule
import torch


class DendriticLossBias(BiasLearningRule):
    def step(self):
        # self.population.bias.data += self.learning_rate * self.population.dendritic_state
        self.population.bias.data += self.learning_rate * self.population.plateau


class DendriticLoss(LearningRule):
    """
    This is the original rule and is gated by plateaus.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.plateau, self.projection.pre.activity)


class DendriticLoss_2(LearningRule):
    """
    This variant 2 is gated by dendritic state. The original rule is gated by plateaus.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)


class DendriticLoss_3(LearningRule):
    """
    This variant 3 is gated by dendritic state. Consults the initial dendritic state and initial activity obtained
    after the forward phase, before comparison to target during the backward phase. Requires BTSP_2 or equivalent rule
    to update the attributes forward_dendritic_state and forward_activity.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.forward_dendritic_state,
                                                   self.projection.pre.forward_activity)


class DendriticLoss_4(LearningRule):
    """
    This variant 4 is gated by dendritic state. The original rule is gated by plateaus.
    This variant returns to learning weights based on activity at the end of the backward phase.
    pre.activity is saturated at 1.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.dendritic_state,
                                                   torch.clamp(self.projection.pre.activity, 0, 1))


class DendriticLoss_5(LearningRule):
    """
    This variant 5 is gated by dendritic state. The original rule is gated by plateaus.
    pre.activity is saturated at 1.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.forward_dendritic_state,
                                                   torch.clamp(self.projection.pre.forward_activity, 0, 1))


class DendriticLoss_6(LearningRule):
    """
    This variant 6 is gated by forward_dendritic_state. pre.forward_activity is saturated at 1.
    """
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(
                torch.clamp(self.projection.post.forward_dendritic_state.detach().clone(), min=-1, max=1),
                torch.clamp(self.projection.pre.forward_activity, min=0, max=1))
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(
                torch.clamp(self.projection.post.forward_dendritic_state.detach().clone(), min=-1, max=1),
                torch.clamp(self.projection.pre.forward_prev_activity, min=0, max=1))
        
        self.projection.weight.data += self.sign * self.learning_rate * delta_weight


class DendriticLoss_7(LearningRule):
    """
    This variant 7 is gated by forward_dendritic_state.
    """
    
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign
    
    def step(self):
        if self.projection.direction in ['forward', 'F']:
            delta_weight = torch.outer(
                torch.clamp(self.projection.post.forward_dendritic_state.detach().clone(), min=-1, max=1),
                self.projection.pre.forward_activity)
        elif self.projection.direction in ['recurrent', 'R']:
            delta_weight = torch.outer(
                torch.clamp(self.projection.post.forward_dendritic_state.detach().clone(), min=-1, max=1),
                self.projection.pre.forward_prev_activity)
        
        self.projection.weight.data += self.sign * self.learning_rate * delta_weight

