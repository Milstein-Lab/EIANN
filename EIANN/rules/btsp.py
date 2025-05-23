from .base_classes import LearningRule, BiasLearningRule
import torch
from EIANN.utils import get_scaled_rectified_sigmoid
import math


class BTSP_19(LearningRule):
    def __init__(self, projection, neg_rate_th=None, dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=1.,
                 temporal_discount=0.25, stochastic=False, learning_rate=None, relu_gate=True):
        """
        This method includes both positive and negative modulatory events. Both positive and negative modulatory events
        nudge the somatic activity.
        Plateaus increment an instructive signal (IS) that decays to zero in two samples. Presynaptic activity
        increments an eligibility trace (ET) that decays to zero in two samples. Positive modulatory events
        result in a BTSP weight update that depends on IS, current weight, and ET.
        Negative modulatory events result in a weight update proportional to plateaus and presynaptic activity.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. During nudging, activities are re-equilibrated across all layers.
        :param projection: :class:'nn.Linear'
        :param neg_rate_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param temporal_discount: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        :param relu_gate: bool
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.neg_rate_th = neg_rate_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.temporal_discount = temporal_discount
        self.stochastic = stochastic
        self.relu_gate = relu_gate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def reinit(self):
        self.projection.pre.past_activity = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.past_plateau = torch.zeros(self.projection.post.size,
                                                   device=self.projection.post.network.device)
    
    def update(self):
        if not self.projection.pre.past_activity_updated:
            if self.projection.direction in ['forward', 'F']:
                self.projection.pre.past_activity = self.projection.pre.activity.detach().clone()
            elif self.projection.direction in ['recurrent', 'R']:
                self.projection.pre.past_activity = self.projection.pre.prev_activity.detach().clone()
            self.projection.pre.past_activity_updated = True
        
        if not self.projection.post.past_plateau_updated:
            self.projection.post.past_plateau = self.projection.post.plateau.detach().clone()
            self.projection.post.past_plateau_updated = True
    
    def step(self):
        # pos error - BTSP weight update
        plateau_prob = self.projection.post.plateau.detach().clone()
        neg_indexes = (self.projection.post.plateau < 0.).nonzero(as_tuple=True)
        plateau_prob[neg_indexes] = 0.
        plateau_prob = plateau_prob.unsqueeze(1)
        
        IS_val = 1.
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.prev_activity, 0., 1.)
            
        # pre activity and post plateau for current sample
        delta_weight = (plateau_prob *
                        ((self.w_max - self.projection.weight) * ET.unsqueeze(0) * IS_val -
                         self.projection.weight * self.dep_ratio *
                         self.q_dep(ET * IS_val).unsqueeze(0))).detach().clone()
        
        # pre activity for prev sample and post plateau for current sample
        past_ET = torch.clamp(self.projection.pre.past_activity, 0., 1.) * self.temporal_discount
        delta_weight += (plateau_prob * ((self.w_max - self.projection.weight) * past_ET.unsqueeze(0) * IS_val -
                                         self.projection.weight * self.dep_ratio *
                                         self.q_dep(past_ET * IS_val).unsqueeze(0))).detach().clone()
        
        # pre activity for current sample and post plateau for prev sample
        past_plateau_prob = self.projection.post.past_plateau.detach().clone()
        neg_indexes = (self.projection.post.past_plateau < 0.).nonzero(as_tuple=True)
        past_plateau_prob[neg_indexes] = 0.
        past_plateau_prob = past_plateau_prob.unsqueeze(1)
        
        past_IS_val = self.temporal_discount
        delta_weight += (past_plateau_prob * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) * past_IS_val -
                                    self.projection.weight * self.dep_ratio *
                                    self.q_dep(ET * past_IS_val).unsqueeze(0))).detach().clone()
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = self.projection.post.plateau.detach().clone()
        neg_error[self.projection.post.plateau > 0.] = 0.
        delta_weight += ET.unsqueeze(0) * neg_error.unsqueeze(1)
        
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_layer_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for post_pop in layer:
            post_pop.prev_activity = post_pop.activity
        
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                # update somatic state and activity
                delta_state = -post_pop.state + post_pop.bias
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
                else:
                    post_pop.activity = post_pop.activation(post_pop.state)
                if store_dynamics:
                    post_pop.backward_steps_activity.append(post_pop.activity.detach().clone())
    
    @classmethod
    def backward_update_layer_dendritic_state(cls, layer):
        """
        Update dendritic state for all populations that receive projections that target the dendritic
        compartment.
        """
        for post_pop in layer:
            if post_pop.backward_projections:
                # update dendritic state
                init_dend_state = False
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment == 'dend':
                        if not init_dend_state:
                            post_pop.dendritic_state = torch.zeros(post_pop.size, device=post_pop.network.device)
                            init_dend_state = True
                        if projection.direction in ['forward', 'F']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            post_pop.dendritic_state = post_pop.dendritic_state + projection(pre_pop.prev_activity)
                if init_dend_state:
                    post_pop.dendritic_state.clamp_(min=-1, max=1)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """
        Integrate top-down inputs and update dendritic state variables.
        :param network:
        :param output:
        :param target:
        :param store_history: bool
        :param store_dynamics: bool
        """
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        output_pop = network.output_pop
        
        # initialize populations that are updated during the backward phase
        for layer in reversed_layers:
            for pop in layer:
                if pop.backward_projections or pop is output_pop:
                    if store_dynamics:
                        pop.backward_steps_activity = []
                # initialize dendritic state variables
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        break
        
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                if t == 0:
                    for pop in layer:
                        for projection in pop:
                            # compute plateau events and nudge somatic state
                            if projection.learning_rule.__class__ == cls:
                                if pop is output_pop:
                                    local_loss = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                    output_pop.dendritic_state = local_loss.detach().clone()
                                    if projection.learning_rule.relu_gate:
                                        local_loss[output_pop.forward_activity == 0.] = 0
                                    output_pop.plateau[:] = local_loss.detach().clone()
                                    output_pop.dend_to_soma[:] = local_loss.detach().clone()
                                else:
                                    max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                    local_loss = pop.dendritic_state.detach().clone()
                                    if projection.learning_rule.relu_gate:
                                        local_loss[pop.forward_activity == 0.] = 0
                                    
                                    pos_avail_indexes = (local_loss > 0.).nonzero().squeeze(1)
                                    if projection.learning_rule.stochastic:
                                        pos_candidate_rel_indexes = (
                                            torch.bernoulli(local_loss[pos_avail_indexes])).nonzero().squeeze(1)
                                    else:
                                        sorted, pos_candidate_rel_indexes = torch.sort(local_loss[pos_avail_indexes],
                                                                                       descending=True, stable=True)
                                    pos_event_indexes = pos_avail_indexes[pos_candidate_rel_indexes][:max_units]
                                    
                                    if projection.learning_rule.neg_rate_th is not None:
                                        local_loss[pop.forward_activity <= projection.learning_rule.neg_rate_th] = 0.
                                    neg_avail_indexes = (local_loss < 0.).nonzero().squeeze(1)
                                    if projection.learning_rule.stochastic:
                                        neg_candidate_rel_indexes = (
                                            torch.bernoulli(-local_loss[neg_avail_indexes])).nonzero().squeeze(1)
                                    else:
                                        sorted, neg_candidate_rel_indexes = torch.sort(local_loss[neg_avail_indexes],
                                                                                       descending=True, stable=True)
                                    neg_event_indexes = neg_avail_indexes[neg_candidate_rel_indexes][-max_units:]
                                    
                                    pop.plateau[pos_event_indexes] = local_loss[pos_event_indexes]
                                    pop.plateau[neg_event_indexes] = local_loss[neg_event_indexes]
                                    pop.dend_to_soma[pos_event_indexes] = local_loss[pos_event_indexes]
                                    pop.dend_to_soma[neg_event_indexes] = local_loss[neg_event_indexes]
                                break
                # update activities
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)
        
        pop_history_stored = []
        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history and pop not in pop_history_stored:
                            pop_history_stored.append(pop)
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        pop.past_plateau_updated = False
                        projection.pre.past_activity_updated = False
                if store_history:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())
