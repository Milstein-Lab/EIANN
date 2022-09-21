import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class LearningRule(object):
    def __init__(self, projection, learning_rate=None):
        self.projection = projection
        if learning_rate is None:
            learning_rate = self.projection.post.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    @classmethod
    def backward(cls, network, output, target):
        pass


class BiasLearningRule(object):
    def __init__(self, population, learning_rate=None):
        self.population = population
        if learning_rate is None:
            learning_rate = self.population.network.learning_rate
        self.learning_rate = learning_rate

    def step(self):
        pass

    @classmethod
    def backward(cls, network, output, target):
        pass


class Backprop(LearningRule):
    @classmethod
    def backward(cls, network, output, target):
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()


class BackpropBias(BiasLearningRule):
    backward = Backprop.backward


class Oja(LearningRule):
    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) - \
                       self.projection.weight * (self.projection.post.activity ** 2).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight


class BCM(LearningRule):
    def __init__(self, projection, theta_init, theta_tau, k, learning_rate=None):
        """

        :param projection:
        :param theta_init: float
        :param theta_tau: float
        :param k: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.theta_tau = theta_tau
        self.k = k
        self.projection.post.theta = torch.ones(projection.post.size) * theta_init

    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity) * \
                       (self.projection.post.activity - self.projection.post.prev_theta).unsqueeze(1)
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward(cls, network, output, target):
        for i, layer in enumerate(network):
            if i > 0:
                for population in layer:
                    for projection in population:
                        if projection.learning_rule_class == cls:
                            population.prev_theta = population.theta.clone()
                            delta_theta = (-population.theta + population.activity ** 2. /
                                           projection.learning_rule.k) / projection.learning_rule.theta_tau
                            population.theta += delta_theta
                            # only update theta once per population
                            break


class GjorgievaHebb(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        delta_weight = torch.outer(self.projection.post.activity, self.projection.pre.activity)
        if self.sign > 0:
            self.projection.weight.data += self.learning_rate * delta_weight
        else:
            self.projection.weight.data -= self.learning_rate * delta_weight


def get_scaled_rectified_sigmoid(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError('scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return np.vectorize(
        lambda xi:
        (target_amp / amp) * (1. / (1. + np.exp(-slope * (max(min(xi, x[-1]), x[0]) - th))) - start_val) + ylim[0])


class BTSP(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """

        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_loss_ET_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_loss_ET_discount = neg_loss_ET_discount
        self.dep_ratio = dep_ratio
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]

    def step(self):
        plateau = self.projection.post.plateau
        discount = plateau.detach().clone()
        discount[plateau > 0] = 1.
        discount[plateau < 0] = self.neg_loss_ET_discount
        IS = torch.abs(plateau).unsqueeze(1)
        ET = torch.outer(discount, self.projection.pre.activity)

        delta_weight = IS * ((self.w_max - self.projection.weight) * ET - \
                       self.projection.weight * self.dep_ratio * self.q_dep(ET))

        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_update_layer_activity(cls, layer):
        """
        Update activity and dendritic state for populations in the layer that receive backward projections.
        :param layer:
        """
        # update activity
        for pop in layer:
            if pop.backward_projections:
                pop.delta_state = torch.zeros(pop.size)
                if hasattr(pop, 'dend_to_soma'):
                    pop.delta_state += pop.dend_to_soma
                for projection in pop.backward_projections:
                    if projection.compartment in [None, 'soma']:
                        pop.delta_state += projection(projection.pre.activity)
                pop.activity = pop.activation(pop.prev_state + pop.delta_state)

        # update dendritic state
        for pop in layer:
            count = 0
            for projection in pop.backward_projections:
                if projection.compartment == 'dend':
                    if count == 0:
                        pop.dendritic_state = torch.zeros(pop.size)
                        count += 1
                    pop.dendritic_state += projection(projection.pre.activity)
            if count > 0:
                pop.dendritic_state = torch.clamp(pop.dendritic_state, min=-1, max=1)

    @classmethod
    def backward(cls, network, output, target):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_layer = reversed_layers.pop(0)
        output_pop = next(iter(output_layer))
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        output_pop.prev_state = output_pop.state.detach().clone()
        output_pop.dend_to_soma = torch.zeros(output_pop.size)
        for projection in output_pop:
            if projection.learning_rule_class == cls:
                output_pop.plateau = torch.zeros(output_pop.size)
                for i in range(output_pop.size):
                    if output_pop.dendritic_state[i] >  projection.learning_rule.pos_loss_th:
                        output_pop.plateau[i] = output_pop.dendritic_state[i]
                        output_pop.dend_to_soma[i] = output_pop.dendritic_state[i]
                        output_pop.activity[i] = \
                            output_pop.activation(output_pop.prev_state[i] + output_pop.dend_to_soma[i])
                    elif output_pop.dendritic_state[i] < projection.learning_rule.neg_loss_th:
                        output_pop.plateau[i] = output_pop.dendritic_state[i]
                break

        # initialize cells that receive backward projections
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections:
                    pop.prev_state = pop.state.detach().clone()
            cls.backward_update_layer_activity(layer)

        # update dendritic state variables
        for layer in reversed_layers:
            for pop in layer:
                for projection in pop.backward_projections:
                    if projection.learning_rule_class == cls:
                        pop.plateau = torch.zeros(pop.size)
                        pop.dend_to_soma = torch.zeros(pop.size)
                        # sort cells by dendritic state
                        _, pop_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                        for i in pop_indexes:
                            if pop.dendritic_state[i] > projection.learning_rule.pos_loss_th:
                                pop.plateau[i] = pop.dendritic_state[i]
                                pop.dend_to_soma[i] = pop.dendritic_state[i]
                                cls.backward_update_layer_activity(layer)
                            elif pop.dendritic_state[i] < projection.learning_rule.neg_loss_th:
                                pop.plateau[i] = pop.dendritic_state[i]
                        break


class DendriticLossBias(BiasLearningRule):
    def step(self):
        # self.population.bias.data += self.learning_rate * self.population.dendritic_state
        self.population.bias.data += self.learning_rate * self.population.plateau

    backward = BTSP.backward


class DendriticLoss(LearningRule):
    def __init__(self, projection, sign=1, learning_rate=None):
        super().__init__(projection, learning_rate)
        self.sign = sign

    def step(self):
        #self.projection.weight.data += self.sign * self.learning_rate * \
        #                               torch.outer(self.projection.post.dendritic_state, self.projection.pre.activity)
        self.projection.weight.data += self.sign * self.learning_rate * \
                                       torch.outer(self.projection.post.plateau, self.projection.pre.activity)

    backward = BTSP.backward








