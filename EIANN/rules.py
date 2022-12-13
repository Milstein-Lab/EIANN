import torch
import torch.nn as nn
import numpy as np
from .utils import get_scaled_rectified_sigmoid
import time


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
                        if projection.learning_rule.__class__ == cls:
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
        pop_set = set()
        for post_pop in layer:
            pop_set.add(post_pop)
            for projection in post_pop.backward_projections:
                pop_set.add(projection.pre)

        for pop in pop_set:
            pop.prev_activity = pop.activity

        # update activity for populations that receive backward projections to the soma
        for pop in layer:
            init_dend_state = False
            if pop.backward_projections:
                # pop.forward_soma_state already contains bias and inputs updated during the forward phase
                delta_state = -pop.state + pop.forward_soma_state
                if hasattr(pop, 'dend_to_soma'):
                    delta_state = delta_state + pop.dend_to_soma
                for projection in pop.backward_projections:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                pop.state = pop.state + delta_state / pop.tau
                pop.activity = pop.activation(pop.state)

        # update dendritic state for populations that receive backward projections to the dendrite
        for pop in layer:
            init_dend_state = False
            for projection in pop.backward_projections:
                pre_pop = projection.pre
                if projection.compartment == 'dend':
                    if not init_dend_state:
                        pop.dendritic_state = torch.zeros(pop.size)
                        init_dend_state = True
                    if projection.direction in ['forward', 'F']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.activity)
                    elif projection.direction in ['recurrent', 'R']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.prev_activity)
            if init_dend_state:
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
        output_pop.forward_soma_state = output_pop.state.detach().clone()
        output_pop.plateau = torch.zeros(output_pop.size)
        output_pop.dend_to_soma = torch.zeros(output_pop.size)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state>  projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.activity[pos_indexes] = \
                    output_pop.activation(output_pop.forward_soma_state[pos_indexes] +
                                          output_pop.dend_to_soma[pos_indexes])
                neg_indexes = (output_pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                # print(output_pop.plateau)
                break

        for layer in reversed_layers:
            # initialize cells that receive backward projections
            for pop in layer:
                if pop.backward_projections:
                    pop.forward_soma_state = pop.state.detach().clone()

            # equilibrate activites and dendritic state variables
            for t in range(network.forward_steps):
                cls.backward_update_layer_activity(layer)
                for pop in layer:
                    for projection in pop.backward_projections:
                        if projection.learning_rule.__class__ == cls:
                            pop.plateau = torch.zeros(pop.size)
                            pop.dend_to_soma = torch.zeros(pop.size)
                            pos_indexes = (pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(
                                as_tuple=True)
                            pop.plateau[pos_indexes] = pop.dendritic_state[pos_indexes]
                            pop.dend_to_soma[pos_indexes] = pop.dendritic_state[pos_indexes]
                            break
            for pop in layer:
                for projection in pop.backward_projections:
                    if projection.learning_rule.__class__ == cls:
                        neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                            as_tuple=True)
                        pop.plateau[neg_indexes] = pop.dendritic_state[neg_indexes]
                        print(pop.plateau)
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


def normalize_weight(projection, scale, autapses=False, axis=1):
    projection.weight.data /= torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    projection.weight.data *= scale
    if not autapses and projection.pre == projection.post:
        for i in range(projection.post.size):
            projection.weight.data[i, i] = 0.







