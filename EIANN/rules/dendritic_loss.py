from .base_classes import LearningRule, BiasLearningRule
import torch


class DendriticLossBias(BiasLearningRule):
    def step(self):
        # self.population.bias.data += self.learning_rate * self.population.dendritic_state
        self.population.bias.data += self.learning_rate * self.population.plateau


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
