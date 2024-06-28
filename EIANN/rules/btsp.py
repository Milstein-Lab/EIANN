from .base_classes import LearningRule, BiasLearningRule
import torch
from EIANN.utils import get_scaled_rectified_sigmoid
import math



class BTSP(LearningRule):
    """
    This original rule variant is not compatible with equilibration of populations in both forward and backward phases.
    """
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')

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
                        pop.dendritic_state = torch.zeros(pop.size, device=pop.network.device)
                        init_dend_state = True
                    if projection.direction in ['forward', 'F']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.activity)
                    elif projection.direction in ['recurrent', 'R']:
                        pop.dendritic_state = pop.dendritic_state + projection(pre_pop.prev_activity)
            if init_dend_state:
                pop.dendritic_state = torch.clamp(pop.dendritic_state, min=-1, max=1)


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
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
                output_pop.forward_soma_state = output_pop.state.detach().clone()
                output_pop.plateau = torch.zeros(output_pop.size, device=network.device)
                output_pop.dend_to_soma = torch.zeros(output_pop.size, device=network.device)
                pos_indexes = (output_pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.activity[pos_indexes] = \
                    output_pop.activation(output_pop.forward_soma_state[pos_indexes] +
                                          output_pop.dend_to_soma[pos_indexes])
                neg_indexes = (output_pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                if store_history:
                    output_pop.append_attribute_history('plateau',output_pop.plateau.detach().clone())
                break
        if store_history:
            output_pop.append_attribute_history('backward_activity', output_pop.activity.detach().clone())

        for layer in reversed_layers:
            # initialize cells that receive backward projections
            for pop in layer:
                if pop.backward_projections:
                    pop.forward_soma_state = pop.state.detach().clone()
                    if store_dynamics:
                        pop.backward_steps_activity = []

            # equilibrate activities and dendritic state variables
            for t in range(network.forward_steps):
                cls.backward_update_layer_activity(layer)
                for pop in layer:
                    if pop.backward_projections:
                        for projection in pop.backward_projections:
                            if projection.learning_rule.__class__ == cls:
                                pop.plateau = torch.zeros(pop.size, device=network.device)
                                pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                                pos_indexes = (pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(
                                    as_tuple=True)
                                pop.plateau[pos_indexes] = pop.dendritic_state[pos_indexes]
                                pop.dend_to_soma[pos_indexes] = pop.dendritic_state[pos_indexes]
                                break
                        if store_dynamics:
                            pop.backward_steps_activity.append(pop.activity.detach().clone())
            for pop in layer:
                if pop.backward_projections:
                    for projection in pop.backward_projections:
                        if projection.learning_rule.__class__ == cls:
                            neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                                as_tuple=True)
                            pop.plateau[neg_indexes] = pop.dendritic_state[neg_indexes]
                            if store_history:
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            break
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_2(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. Only positive
        modulatory events nudge the somatic activity. This variant is tolerant to the update_phase of projections
        being specified as 'A' or 'All', whereas the original class could only update projections either in the
        forward or backward phase, but not both. In variant 2, plateaus can only occur in a specified maximum fraction
        of units in each layer. The entire layer is re-equilibrated after plateaus are selected.
        :param projection: :class:'nn.Linear'
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_loss_ET_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_loss_ET_discount = neg_loss_ET_discount
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')

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
    def backward_equilibrate_layer_activity(cls, layer, forward_steps, store_dynamics=False):
        """
        Equilibrate somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param forward_steps: int
        :param store_dynamics: bool
        """
        for t in range(forward_steps):
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop

        if store_dynamics:
            # initialize populations that are updated during the backward phase
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        pop.backward_steps_activity = []
        
        # compute plateau events and nudge somatic state
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        output_pop.plateau = torch.zeros(output_pop.size, device=network.device)
        output_pop.dend_to_soma = torch.zeros(output_pop.size, device=network.device)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state >
                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                neg_indexes = (output_pop.dendritic_state <
                               projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                # output_pop.dend_to_soma[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                break

        # re-equilibrate soma states and activities
        cls.backward_equilibrate_layer_activity(output_layer, network.forward_steps, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)

                        sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                        pos_indexes = (pop.dendritic_state[sorted_indexes] >
                                       projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)[0][:max_units]
                        pos_plateau_indexes = sorted_indexes[pos_indexes]
                        pop.plateau[pos_plateau_indexes] = pop.dendritic_state[pos_plateau_indexes]
                        pop.dend_to_soma[pos_plateau_indexes] = 1.  # pop.dendritic_state[pos_plateau_indexes]

                        neg_indexes = (pop.dendritic_state[sorted_indexes] <
                                       projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)[0][-max_units:]
                        neg_plateau_indexes = sorted_indexes[neg_indexes]
                        pop.plateau[neg_plateau_indexes] = pop.dendritic_state[neg_plateau_indexes]
                        # pop.dend_to_soma[neg_plateau_indexes] = pop.dendritic_state[neg_plateau_indexes]

                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
            cls.backward_equilibrate_layer_activity(layer, network.forward_steps, store_dynamics=store_dynamics)

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_3(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. Only positive
        modulatory events nudge the somatic activity. This variant is tolerant to the update_phase of projections
        being specified as 'A' or 'All', whereas the original class could only update projections either in the
        forward or backward phase, but not both.
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')

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
                if hasattr(post_pop, 'dend_to_soma'):
                    delta_state = delta_state + post_pop.dend_to_soma
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop
        
        if store_dynamics:
            # initialize populations that are updated during the backward phase
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        pop.backward_steps_activity = []

        # compute plateau events and nudge somatic state
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        output_pop.plateau = torch.zeros(output_pop.size, device=network.device)
        output_pop.dend_to_soma = torch.zeros(output_pop.size, device=network.device)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state >
                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                neg_indexes = (output_pop.dendritic_state <
                               projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                # output_pop.dend_to_soma[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                break
        # re-equilibrate soma states and activities
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            for t in range(network.forward_steps):
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            pop.plateau = torch.zeros(pop.size, device=network.device)
                            pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                            pos_indexes = (pop.dendritic_state > projection.learning_rule.pos_loss_th).nonzero(
                                as_tuple=True)
                            pop.plateau[pos_indexes] = pop.dendritic_state[pos_indexes]
                            pop.dend_to_soma[pos_indexes] = 1.  # pop.dendritic_state[pos_indexes]
                            break
            for pop in layer:
                for projection in pop.backward_projections:
                    if projection.learning_rule.__class__ == cls:
                        neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                            as_tuple=True)
                        pop.plateau[neg_indexes] = pop.dendritic_state[neg_indexes]
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                        break

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_4(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. Only positive
        modulatory events nudge the somatic activity. This variant is tolerant to the update_phase of projections
        being specified as 'A' or 'All', whereas the original class could only update projections either in the
        forward or backward phase, but not both. In variant 4, units are selected for positive modulatory events in
        order of largest loss, one at a time, and the entire layer is re-equilibrated after each event.
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
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')

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
                if hasattr(post_pop, 'dend_to_soma'):
                    delta_state = delta_state + post_pop.dend_to_soma
                for projection in post_pop:
                    pre_pop = projection.pre
                    if projection.compartment in [None, 'soma']:
                        if projection.direction in ['forward', 'F']:
                            delta_state = delta_state + projection(pre_pop.activity)
                        elif projection.direction in ['recurrent', 'R']:
                            delta_state = delta_state + projection(pre_pop.prev_activity)
                post_pop.state = post_pop.state + delta_state / post_pop.tau
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop

        if store_dynamics:
            # initialize populations that are updated during the backward phase
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        pop.backward_steps_activity = []
        
        # compute plateau events and nudge somatic state
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        output_pop.plateau = torch.zeros(output_pop.size, device=network.device)
        output_pop.dend_to_soma = torch.zeros(output_pop.size, device=network.device)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state >
                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                neg_indexes = (output_pop.dendritic_state <
                               projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                # output_pop.dend_to_soma[neg_indexes] = output_pop.dendritic_state[neg_indexes]
                break

        # re-equilibrate soma states and activities
        for t in range(network.forward_steps):
            cls.backward_update_layer_activity(output_layer, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        pos_plateau_indexes = []
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)
                        # sort cells by dendritic state
                        while len(pos_plateau_indexes) < pop.size:
                            sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                            pos_indexes = (pop.dendritic_state[sorted_indexes] >
                                           projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)[0]
                            sorted_pos_indexes = sorted_indexes[pos_indexes]
                            avail_indexes = \
                                torch.where(torch.isin(sorted_pos_indexes, torch.tensor(pos_plateau_indexes),
                                                       invert=True))[0]
                            if len(avail_indexes) > 0:
                                pos_plateau_index = sorted_pos_indexes[avail_indexes[0]]
                            else:
                                break
                            pos_plateau_indexes.append(pos_plateau_index)
                            pop.plateau[pos_plateau_index] = pop.dendritic_state[pos_plateau_index]
                            pop.dend_to_soma[pos_plateau_index] = 1.  # pop.dendritic_state[pos_indexes]
                            for t in range(network.forward_steps):
                                cls.backward_update_layer_activity(layer, store_history)
                            cls.backward_update_layer_dendritic_state(layer)
                        neg_indexes = (pop.dendritic_state < projection.learning_rule.neg_loss_th).nonzero(
                            as_tuple=True)[0]
                        avail_indexes = \
                            torch.where(torch.isin(neg_indexes, torch.tensor(pos_plateau_indexes), invert=True))[0]
                        if len(avail_indexes) > 0:
                            neg_plateau_indexes = neg_indexes[avail_indexes]
                            pop.plateau[neg_plateau_indexes] = pop.dendritic_state[neg_plateau_indexes]
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                        break

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_5(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, temporal_discount=0.25, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Unlike the original BTSP class, this method utilizes only positive modulatory events. Positive
        modulatory events nudge the somatic activity. However, now presynaptic eligibility and postsynaptic instruction
        persist and decay from the current sample to the next sample.
        This variant is tolerant to the update_phase of projections being specified as 'A' or 'All', whereas the
        original class could only update projections either in the forward or backward phase, but not both.
        Like variant 2, plateaus can only occur in a specified maximum fraction of units in each layer. The entire
        layer is re-equilibrated after plateaus are selected.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param temporal_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.temporal_discount = temporal_discount
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')

    def reinit(self):
        self.projection.pre.past_activity = torch.zeros(self.projection.pre.size,
                                                        device=self.projection.pre.network.device)
        self.projection.post.past_plateau = torch.zeros(self.projection.post.size,
                                                        device=self.projection.post.network.device)

    def update(self):
        self.projection.pre.past_activity = self.projection.pre.activity.detach().clone()
        self.projection.post.past_plateau = self.projection.post.plateau.detach().clone()

    def step(self):
        if torch.any(self.projection.post.plateau > 0):
            any_IS_now = True
            IS = torch.unsqueeze(self.projection.post.plateau, 1)
        else:
            any_IS_now = False

        ET = self.projection.pre.activity

        if any_IS_now:
            past_ET = self.projection.pre.past_activity * self.temporal_discount
            past_delta_weight = IS * (
                    (self.w_max - self.projection.weight) * torch.unsqueeze(past_ET, 0) -
                    self.projection.weight * self.dep_ratio * torch.unsqueeze(self.q_dep(past_ET), 0))
            self.projection.weight.data += self.learning_rate * past_delta_weight

        if torch.any(self.projection.post.past_plateau > 0):
            past_IS = torch.unsqueeze(self.projection.post.past_plateau, 1)
            future_ET = self.projection.pre.activity * self.temporal_discount
            future_delta_weight = past_IS * (
                    (self.w_max - self.projection.weight) * torch.unsqueeze(future_ET, 0) -
                    self.projection.weight * self.dep_ratio * torch.unsqueeze(self.q_dep(future_ET), 0))
            self.projection.weight.data += self.learning_rate * future_delta_weight

        if any_IS_now:
            delta_weight = IS * (
                    (self.w_max - self.projection.weight) * torch.unsqueeze(ET, 0) -
                    self.projection.weight * self.dep_ratio * torch.unsqueeze(self.q_dep(ET), 0))
            self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_equilibrate_layer_activity(cls, layer, forward_steps, store_dynamics=False):
        """
        Equilibrate somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param forward_steps: int
        :param store_dynamics: bool
        """
        for t in range(forward_steps):
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop
        
        # compute plateau events and nudge somatic state
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        output_pop.plateau = torch.zeros(output_pop.size, device=network.device)
        output_pop.dend_to_soma = torch.zeros(output_pop.size, device=network.device)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state >
                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                break

        # re-equilibrate soma states and activities
        cls.backward_equilibrate_layer_activity(output_layer, network.forward_steps, store_dynamics=store_dynamics)
        if store_history:
            output_pop.append_attribute_history('plateau', output_pop.plateau.detach().clone())

        for layer in reversed_layers:
            # equilibrate activities and dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                        pop.plateau = torch.zeros(pop.size, device=network.device)
                        pop.dend_to_soma = torch.zeros(pop.size, device=network.device)

                        sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                        pos_indexes = (pop.dendritic_state[sorted_indexes] >
                                       projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)[0][:max_units]
                        pos_plateau_indexes = sorted_indexes[pos_indexes]
                        pop.plateau[pos_plateau_indexes] = pop.dendritic_state[pos_plateau_indexes]
                        pop.dend_to_soma[pos_plateau_indexes] = 1.  # pop.dendritic_state[pos_plateau_indexes]

                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
            cls.backward_equilibrate_layer_activity(layer, network.forward_steps, store_dynamics=store_dynamics)

        if store_history:
            for layer in network:
                for pop in layer:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_6(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Unlike the original BTSP class, this method utilizes only positive modulatory events. Positive
        modulatory events nudge the somatic activity. However, now presynaptic eligibility and postsynaptic instruction
        persist and decay across samples with a timescale set by a temporal_decay hyperparameter.
        Previous versions of the BTSP class considered only the ET in determining the sign of delta_weight. This
        effectively presumed that IS = 1, and ET * IS = ET. However, plateau amplitude varied with error, and was used
        to scale the learning_rate.
        In this BTSP class variant 6, ET * IS is consulted to determine change in weight. Both ET and IS saturate at 1,
        and accumulate or decay across samples. Plateau amplitude no longer separately modulates the learning_rate.
        In previous versions of the BTSP class, plateaus influenced somatic activity by slowly equilibrating the
        cell's internal state. In this variant, plateaus instantaneously increment firing rate.
        This variant is tolerant to the update_phase of projections being specified as 'A' or 'All', whereas the
        original class could only update projections either in the forward or backward phase, but not both.
        Like variant 2, plateaus can only occur in a specified maximum fraction of units in each layer. The entire
        layer is re-equilibrated after plateaus are selected.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        projection.post.register_attribute_history('ET')
        projection.post.register_attribute_history('IS')

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):

        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_equilibrate_layer_activity(cls, layer, forward_steps, store_dynamics=False):
        """
        Equilibrate somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param forward_steps: int
        :param store_dynamics: bool
        """
        for t in range(forward_steps):
            cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)

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
                post_pop.activity = post_pop.activation(post_pop.state)
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.activity = post_pop.activity + post_pop.dend_to_soma
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
        output_layer = reversed_layers.pop(0)
        output_pop = network.output_pop
        
        # initialize populations that are updated during the backward phase
        for layer in network:
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
        
        # compute plateau events and nudge somatic state
        output_pop.dendritic_state = torch.clamp(target - output, min=-1, max=1)
        for projection in output_pop:
            if projection.learning_rule.__class__ == cls:
                pos_indexes = (output_pop.dendritic_state >
                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                break
        # re-equilibrate soma states and activities
        cls.backward_equilibrate_layer_activity(output_layer, network.forward_steps, store_dynamics=store_dynamics)
        
        for layer in reversed_layers:
            for t in range(network.forward_steps):
                # equilibrate activities and dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                            if pop.plateau.count_nonzero() < max_units:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True, stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
                cls.backward_update_layer_activity(layer, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_7(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Unlike the original BTSP class, this method utilizes only positive modulatory events. Positive
        modulatory events nudge the somatic activity. However, now presynaptic eligibility and postsynaptic instruction
        persist and decay across samples with a timescale set by a temporal_decay hyperparameter.
        Previous versions of the BTSP class considered only the ET in determining the sign of delta_weight. This
        effectively presumed that IS = 1, and ET * IS = ET. However, plateau amplitude varied with error, and was used
        to scale the learning_rate.
        In this BTSP class variant 6, ET * IS is consulted to determine change in weight. Both ET and IS saturate at 1,
        and accumulate or decay across samples. Plateau amplitude no longer separately modulates the learning_rate.
        In previous versions of the BTSP class, plateaus influenced somatic activity by slowly equilibrating the
        cell's internal state. In this variant, plateaus instantaneously increment firing rate.
        This variant is tolerant to the update_phase of projections being specified as 'A' or 'All', whereas the
        original class could only update projections either in the forward or backward phase, but not both.
        Like variant 2, plateaus can only occur in a specified maximum fraction of units in each layer.
        While variants 2 and 6 equilibrates each layer sequentially and independently after plateaus were selected, this
        variant 7 selects plateaus and equilibrates activity in all layers simultaneously.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        projection.post.register_attribute_history('ET')
        projection.post.register_attribute_history('IS')

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):

        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                if pop.plateau.count_nonzero() < max_units:
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                        stable=True)
                                    candidate_indexes = \
                                        ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                         (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                    if len(candidate_indexes) > 0:
                                        plateau_index = sorted_indexes[candidate_indexes][0]
                                        pop.plateau[plateau_index] = 1.
                                        pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_8(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, anti_hebb_th=0.1, learning_rate=None):
        """
        Like variant 7, except now an anti-Hebbian depression occurs in cells that are active and receiving positive
        feedback within a range less than plateau threshold.
        ** There is an error in this implementation. In hidden layer E cells, the anti_hebb_th was compared to
        firing rate, when it should have been compared to dendritic_state.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param anti_hebb_th: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.anti_hebb_th = anti_hebb_th
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        projection.post.register_attribute_history('ET')
        projection.post.register_attribute_history('IS')

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        delta_weight = -self.learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                if pop.plateau.count_nonzero() < max_units:
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                        stable=True)
                                    candidate_indexes = \
                                        ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                         (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                    if len(candidate_indexes) > 0:
                                        plateau_index = sorted_indexes[candidate_indexes][0]
                                        pop.plateau[plateau_index] = 1.
                                        pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > projection.learning_rule.anti_hebb_th))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_9(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01, dep_width=0.01,
                 anti_hebb_th=0.1, learning_rate=None):
        """
        Similar to variant 8. Anti-Hebbian depression is triggered when an active unit has a dendritic state that
        crosses a threshold. This variant 9 does not impose a hard max_pop_fraction on plateaus.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param anti_hebb_th: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.anti_hebb_th = anti_hebb_th
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        projection.post.register_attribute_history('ET')
        projection.post.register_attribute_history('IS')

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        # TODO: Should these activities be clamped?
        delta_weight = -self.learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                    stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > 0) &
                                     (pop.dendritic_state > projection.learning_rule.anti_hebb_th))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_10(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01, dep_width=0.01,
                 anti_hebb_th=(-0.1, 0.1), learning_rate=None, anti_hebb_learning_rate=None):
        """
        Similar to variant 9. Anti-Hebbian depression is triggered when an active unit has a dendritic state that
        is between a min and max bound.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param anti_hebb_th: tuple of float
        :param learning_rate: float
        :param anti_hebb_learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.anti_hebb_th = anti_hebb_th
        if anti_hebb_learning_rate is None:
            self.anti_hebb_learning_rate = self.learning_rate
        else:
            self.anti_hebb_learning_rate = anti_hebb_learning_rate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        projection.post.register_attribute_history('ET')
        projection.post.register_attribute_history('IS')

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def update(self):
        self.projection.pre.ET *= (1. - 1./self.decay_tau)
        self.projection.post.IS *= (1. - 1./self.decay_tau)
        self.projection.pre.ET_updated = False
        self.projection.post.IS_updated = False

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        # TODO: Should these activities be clamped?
        delta_weight = -self.anti_hebb_learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                    stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_updated:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_updated = True
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th[1]))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > 0) &
                                     (pop.dendritic_state > projection.learning_rule.anti_hebb_th[0]) &
                                     (pop.dendritic_state <= projection.learning_rule.anti_hebb_th[1]))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_updated:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_updated = True
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_11(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, decay_tau=2, dep_ratio=1., dep_th=0.01, dep_width=0.01,
                 anti_hebb_th=(-0.1, 0.1), learning_rate=None, anti_hebb_learning_rate=None, refractory=1):
        """
        Similar to variant 10. Anti-Hebbian depression is triggered when an active unit has a dendritic state that
        is between a min and max bound. Cells cannot plateau within a refractory period.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param decay_tau: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param anti_hebb_th: tuple of float
        :param learning_rate: float
        :param anti_hebb_learning_rate: float
        :param refractory: int
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.decay_tau = decay_tau
        self.dep_ratio = dep_ratio
        self.anti_hebb_th = anti_hebb_th
        if anti_hebb_learning_rate is None:
            self.anti_hebb_learning_rate = self.learning_rate
        else:
            self.anti_hebb_learning_rate = anti_hebb_learning_rate
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        self.refractory = refractory
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
        projection.post.register_attribute_history('ET')
        projection.post.register_attribute_history('IS')

    def reinit(self):
        self.projection.pre.ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.IS = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        self.projection.post.plateau_refractory = (
            torch.zeros(self.projection.post.size, device=self.projection.post.network.device))
        self.projection.pre.ET_incremented = False
        self.projection.post.IS_incremented = False
        self.projection.pre.ET_decayed = False
        self.projection.post.IS_decayed = False

    def update(self):
        if not self.projection.pre.ET_decayed:
            self.projection.pre.ET *= (1. - 1./self.decay_tau)
            self.projection.pre.ET_decayed = True
            self.projection.pre.ET_incremented = False
        if not self.projection.post.IS_decayed:
            self.projection.post.IS *= (1. - 1./self.decay_tau)
            self.projection.post.IS_decayed = True
            self.projection.post.IS_incremented = False
            self.projection.post.plateau_refractory[self.projection.post.plateau == 1] += 1
            self.projection.post.plateau_refractory[self.projection.post.plateau_refractory >= self.refractory] = 0

    def step(self):
        # BTSP
        ET_IS = torch.outer(self.projection.post.IS, self.projection.pre.ET)
        delta_weight = self.learning_rate * (
                (self.w_max - self.projection.weight) * ET_IS -
                self.projection.weight * self.dep_ratio * self.q_dep(ET_IS))
        self.projection.weight.data += delta_weight

        # anti-Hebbian depression
        post_activity = torch.zeros(self.projection.post.size, device=self.projection.post.network.device)
        meets_criterion = self.projection.post.meets_BTSP_anti_hebb_criterion
        post_activity[meets_criterion] = self.projection.post.activity[meets_criterion]
        # TODO: Should these activities be clamped?
        delta_weight = -self.anti_hebb_learning_rate * torch.outer(post_activity, self.projection.pre.activity)
        self.projection.weight.data += delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    post_pop.activity = post_pop.activation(post_pop.state)
                    if hasattr(post_pop, 'dend_to_soma'):
                        post_pop.activity = post_pop.activity + post_pop.dend_to_soma
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                output_pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                pos_indexes = (output_pop.dendritic_state >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                output_pop.plateau[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                                output_pop.dend_to_soma[pos_indexes] = 1.  # output_pop.dendritic_state[pos_indexes]
                            else:
                                sorted, sorted_indexes = torch.sort(pop.dendritic_state, descending=True,
                                                                    stable=True)
                                candidate_indexes = \
                                    ((pop.dendritic_state[sorted_indexes] > projection.learning_rule.pos_loss_th) &
                                     (pop.plateau[sorted_indexes] == 0) &
                                     (pop.plateau_refractory < projection.learning_rule.refractory)).nonzero().squeeze(1)
                                if len(candidate_indexes) > 0:
                                    plateau_index = sorted_indexes[candidate_indexes][0]
                                    pop.plateau[plateau_index] = 1.
                                    pop.dend_to_soma[plateau_index] = 1.  # pop.dendritic_state[pos_plateau_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if not pop.IS_incremented:
                            pop.IS += pop.plateau
                            pop.IS.clamp_(0., 1.)
                            pop.IS_incremented = True
                            pop.IS_decayed = False
                            if pop is output_pop:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.dendritic_state < -projection.learning_rule.anti_hebb_th[1]))
                            else:
                                pop.meets_BTSP_anti_hebb_criterion = \
                                    ((pop.plateau == 0) &
                                     (pop.activity > 0) &
                                     (pop.dendritic_state > projection.learning_rule.anti_hebb_th[0]) &
                                     (pop.dendritic_state <= projection.learning_rule.anti_hebb_th[1]))
                            if store_history:
                                pop.append_attribute_history('IS', pop.IS.detach().clone())
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                        if not projection.pre.ET_incremented:
                            projection.pre.ET += projection.pre.activity
                            projection.pre.ET.clamp_(0., 1.)
                            projection.pre.ET_incremented = True
                            projection.pre.ET_decayed = False
                            if store_history:
                                projection.pre.append_attribute_history('ET', projection.pre.ET.detach().clone())
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_12_cont(LearningRule):
    def __init__(self, projection, dep_ratio=1., dep_th=0.01, dep_width=0.01, learning_rate=None):
        """
        This variant is a stripped-down version of BTSP with no thresholds for mod events. Negative mod events
        evoke a linear weight-dependent depression.
        :param projection: :class:'nn.Linear'
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.dep_ratio = dep_ratio
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')

    def step(self):
        # BTSP
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        unsqueezed_ET = ET.unsqueeze(0)
        delta_weight = IS * ((self.w_max - self.projection.weight) * unsqueezed_ET -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # negative dendritic state triggers linear weight-dependent depression
        neg_error = plateau.detach().clone()
        neg_error[plateau > 0.] = 0.
        neg_error = neg_error.unsqueeze(1)
        delta_weight = neg_error * self.projection.weight * unsqueezed_ET
        self.projection.weight.data += self.learning_rate * delta_weight

    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)

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
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                pop.plateau = pop.dendritic_state.detach().clone()
                                pop.dend_to_soma = pop.dendritic_state.detach().clone()
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)

        for layer in network:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        if store_history:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                        break
                if pop.backward_projections or pop is output_pop:
                    if store_history:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_13(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_loss_ET_discount=0.25,
                 dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity. Plateaus can only occur in a
        specified maximum fraction of units in each layer. All layers are equilibrated simultaneously.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_loss_ET_discount: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_loss_ET_discount = neg_loss_ET_discount
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        plateau = self.projection.post.plateau
        discount = plateau.detach().clone()
        discount[plateau > 0] = 1.
        discount[plateau < 0] = self.neg_loss_ET_discount
        IS = torch.abs(plateau).unsqueeze(1)
        ET = torch.outer(discount, self.projection.pre.activity)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET))
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
            for post_pop in layer:
                if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                    # update somatic state and activity
                    delta_state = -post_pop.state + post_pop.bias
                    if hasattr(post_pop, 'dend_to_soma'):
                        delta_state = delta_state + post_pop.dend_to_soma
                    for projection in post_pop:
                        pre_pop = projection.pre
                        if projection.compartment in [None, 'soma']:
                            if projection.direction in ['forward', 'F']:
                                delta_state = delta_state + projection(pre_pop.activity)
                            elif projection.direction in ['recurrent', 'R']:
                                delta_state = delta_state + projection(pre_pop.prev_activity)
                    post_pop.state = post_pop.state + delta_state / post_pop.tau
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
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
        
        # compute plateau events and nudge somatic state
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                pop.dendritic_state = torch.clamp(target - output_pop.activity, min=-1, max=1)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                pos_indexes = (pop.dendritic_state[avail_indexes] >
                                               projection.learning_rule.pos_loss_th).nonzero(as_tuple=True)
                                pos_avail_indexes = avail_indexes[pos_indexes]
                                neg_indexes = (pop.dendritic_state[avail_indexes] <
                                               projection.learning_rule.neg_loss_th).nonzero(as_tuple=True)
                                neg_avail_indexes = avail_indexes[neg_indexes]
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                if pos_remaining > 0 or neg_remaining > 0:
                                    sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                        descending=True, stable=True)
                                    sorted_avail_indexes = avail_indexes[sorted_indexes]
                                if pos_remaining > 0:
                                    pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                   projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                    pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                else:
                                    pos_avail_indexes = []
                                if neg_remaining > 0:
                                    neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                   projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                    neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                else:
                                    neg_avail_indexes = []
                            pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                            pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                            pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        if store_history:
            for layer in network:
                for pop in layer:
                    for projection in pop:
                        if projection.learning_rule.__class__ == cls:
                            if store_history:
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                            break
                    if pop.backward_projections or pop is output_pop:
                        if store_history:
                            if store_dynamics:
                                pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                            else:
                                pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_14(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, dep_ratio=1., dep_th=0.01,
                 dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity. Plateaus can only occur in a
        specified maximum fraction of units in each layer. All layers are equilibrated simultaneously.
        Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration.
        Negative modulatory events result in a weight update proportional to dendritic state and presynaptic activity.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        # pos error - BTSP weight update
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = plateau.detach().clone()
        neg_error[plateau > 0.] = 0.
        delta_weight = torch.outer(neg_error, ET)
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
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
        
        # compute plateau events and nudge somatic state
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                if len(avail_indexes) > 0:
                                    pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                    neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                    if pos_remaining > 0 or neg_remaining > 0:
                                        sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                            descending=True, stable=True)
                                        sorted_avail_indexes = avail_indexes[sorted_indexes]
                                    if pos_remaining > 0:
                                        pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                       projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                        pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                    else:
                                        pos_avail_indexes = []
                                    if neg_remaining > 0:
                                        neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                       projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                        neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                    else:
                                        neg_avail_indexes = []
                                    pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                                    pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        if store_history:
            for layer in network:
                for pop in layer:
                    for projection in pop:
                        if projection.learning_rule.__class__ == cls:
                            if store_history:
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                            break
                    if pop.backward_projections or pop is output_pop:
                        if store_history:
                            if store_dynamics:
                                pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                            else:
                                pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_15(LearningRule):
    def __init__(self, projection, pos_loss_th=2.440709E-01, neg_loss_th=-4.592181E-01, neg_rate_th=None, dep_ratio=1.,
                 dep_th=0.01, dep_width=0.01, max_pop_fraction=0.025, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity. Plateaus can only occur in a
        specified maximum fraction of units in each layer. All layers are equilibrated simultaneously.
        Nudges to somatic state are applied instantaneously, rather than being subject to slow equilibration.
        Negative modulatory events only occur when neuronal firing rate is below a threshold.
        Negative modulatory events result in a weight update proportional to dendritic state and presynaptic activity.
        :param projection: :class:'nn.Linear'
        :param pos_loss_th: float
        :param neg_loss_th: float
        :param neg_rate_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.pos_loss_th = pos_loss_th
        self.neg_loss_th = neg_loss_th
        self.neg_rate_th = neg_rate_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        # pos error - BTSP weight update
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = plateau.detach().clone()
        if self.neg_rate_th is None:
            neg_error[plateau > 0.] = 0.
        else:
            # TODO: This will veto negative mod events when activity exceeds a threshold, should be opposite
            neg_error[(plateau > 0.) | (self.projection.post.activity > self.neg_rate_th)] = 0.
        delta_weight = torch.outer(neg_error, ET)
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_update_activity(cls, network, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive projections with update_phase in
        ['B', 'backward', 'A', 'all'].
        :param layer:
        :param store_dynamics: bool
        """
        for layer in network:
            for post_pop in layer:
                post_pop.prev_activity = post_pop.activity
        for layer in network:
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
                    post_pop.dendritic_state = torch.clamp(post_pop.dendritic_state, min=-1, max=1)
    
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
        
        # compute plateau events and nudge somatic state
        for t in range(network.forward_steps):
            for layer in reversed_layers:
                # update dendritic state variables
                cls.backward_update_layer_dendritic_state(layer)
                for pop in layer:
                    for projection in pop:
                        # compute plateau events and nudge somatic state
                        if projection.learning_rule.__class__ == cls:
                            if pop is output_pop:
                                if t == 0:
                                    output_pop.dendritic_state = (
                                        torch.clamp(target - output_pop.activity, min=-1, max=1))
                                    output_pop.plateau = output_pop.dendritic_state.detach().clone()
                                    output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                            else:
                                max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                                avail_indexes = (pop.plateau == 0).nonzero().squeeze(1)
                                if len(avail_indexes) > 0:
                                    pos_remaining = max_units - (pop.plateau > 0).count_nonzero()
                                    neg_remaining = max_units - (pop.plateau < 0).count_nonzero()
                                    if pos_remaining > 0 or neg_remaining > 0:
                                        sorted, sorted_indexes = torch.sort(pop.dendritic_state[avail_indexes],
                                                                            descending=True, stable=True)
                                        sorted_avail_indexes = avail_indexes[sorted_indexes]
                                    if pos_remaining > 0:
                                        pos_indexes = (pop.dendritic_state[sorted_avail_indexes] >
                                                       projection.learning_rule.pos_loss_th).nonzero().squeeze(1)
                                        pos_avail_indexes = sorted_avail_indexes[pos_indexes][:pos_remaining]
                                    else:
                                        pos_avail_indexes = []
                                    if neg_remaining > 0:
                                        neg_indexes = (pop.dendritic_state[sorted_avail_indexes] <
                                                       projection.learning_rule.neg_loss_th).nonzero().squeeze(1)
                                        neg_avail_indexes = sorted_avail_indexes[neg_indexes][-neg_remaining:]
                                    else:
                                        neg_avail_indexes = []
                                    pop.plateau[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.dend_to_soma[pos_avail_indexes] = pop.dendritic_state[pos_avail_indexes]
                                    pop.plateau[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                                    pop.dend_to_soma[neg_avail_indexes] = pop.dendritic_state[neg_avail_indexes]
                            break
            # update activities
            cls.backward_update_activity(network, store_dynamics=store_dynamics)
        
        if store_history:
            for layer in network:
                for pop in layer:
                    for projection in pop:
                        if projection.learning_rule.__class__ == cls:
                            if store_history:
                                pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                                pop.append_attribute_history('backward_dendritic_state',
                                                             pop.dendritic_state.detach().clone())
                            break
                    if pop.backward_projections or pop is output_pop:
                        if store_history:
                            if store_dynamics:
                                pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                            else:
                                pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_16(LearningRule):
    def __init__(self, projection, neg_rate_th=None, dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=0.025,
                 stochastic=True, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity. Positive modulatory events
        result in a BTSP weight update that depends on plateaus, current weight, and (forward) presynaptic activity.
        Negative modulatory events only occur when postsynaptic activity is above a threshold. Negative modulatory
        events result in a weight update proportional to plateaus and (forward) presynaptic activity.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. Nudged activity is only passed top-down, but not laterally, with no
        equilibration.
        :param projection: :class:'nn.Linear'
        :param neg_rate_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.neg_rate_th = neg_rate_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.stochastic = stochastic
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def step(self):
        # pos error - BTSP weight update
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.forward_activity, 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.forward_prev_activity, 0., 1.)
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = plateau.detach().clone()
        neg_error[plateau > 0.] = 0.
        delta_weight = torch.outer(neg_error, ET)
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_nudge_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive a nudge.
        :param layer:
        :param store_dynamics: bool
        """
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.prev_activity = post_pop.activity
                    post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
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
        
        for layer in reversed_layers:
            # update dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        if pop is output_pop:
                            output_pop.dendritic_state = (
                                torch.clamp(target - output_pop.activity, min=-1, max=1))
                            output_pop.plateau = output_pop.dendritic_state.detach().clone()
                            output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                        else:
                            max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                            local_loss = pop.dendritic_state.detach().clone()
                            
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
                            # if projection.learning_rule.neg_rate_th is not None:
                            #     neg_event_rel_indexes = (pop.forward_activity[neg_candidate_indexes] >
                            #                              projection.learning_rule.neg_rate_th).nonzero().squeeze(1)
                            #     neg_event_indexes = neg_candidate_indexes[neg_event_rel_indexes]
                            # else:
                            #     neg_event_indexes = neg_candidate_indexes
                            pop.plateau[pos_event_indexes] = pop.dendritic_state[pos_event_indexes]
                            pop.plateau[neg_event_indexes] = pop.dendritic_state[neg_event_indexes]
                            pop.dend_to_soma[pos_event_indexes] = pop.dendritic_state[pos_event_indexes]
                            pop.dend_to_soma[neg_event_indexes] = pop.dendritic_state[neg_event_indexes]
                        break
            # update activities
            cls.backward_nudge_activity(layer, store_dynamics=store_dynamics)
        
        if store_history:
            for layer in network:
                for pop in layer:
                    for projection in pop:
                        if projection.learning_rule.__class__ == cls:
                            pop.append_attribute_history('plateau', pop.plateau.detach().clone())
                            pop.append_attribute_history('backward_dendritic_state',
                                                         pop.dendritic_state.detach().clone())
                            break
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_17(LearningRule):
    def __init__(self, projection, neg_rate_th=None, dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=1.,
                 temporal_discount=0.25, stochastic=False, learning_rate=None):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity.
        Plateaus increment an instructive signal (IS) that decays in two time steps. (Forward) presynaptic activity
        increments an eligibility trace (ET) that decays to zero in two time steps. Positive modulatory events
        result in a BTSP weight update that depends on IS, current weight, and ET.
        Negative modulatory events only occur when postsynaptic activity is above a threshold. Negative modulatory
        events result in a weight update proportional to plateaus and (forward) presynaptic activity.
        If unit selection is stochastic, hidden units are selected for a weight update with a probability proportional
        to dendritic state by sampling a Bernoulli distribution. Otherwise, units are sorted by dendritic state. A fixed
        maximum fraction of hidden units are updated at each train step. Hidden units are "nudged" by dendritic state
        when selected for a weight update. Nudges to somatic state are applied instantaneously, rather than being
        subject to slow equilibration. Nudged activity is only passed top-down, but not laterally, with no
        equilibration.
        :param projection: :class:'nn.Linear'
        :param neg_rate_th: float
        :param dep_ratio: float
        :param dep_th: float
        :param dep_width: float
        :param max_pop_fraction: float in [0, 1]
        :param temporal_discount: float in [0, 1]
        :param stochastic: bool
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        self.q_dep = get_scaled_rectified_sigmoid(dep_th, dep_th + dep_width)
        self.neg_rate_th = neg_rate_th
        self.dep_ratio = dep_ratio
        self.max_pop_fraction = max_pop_fraction
        self.temporal_discount = temporal_discount
        self.stochastic = stochastic
        if self.projection.weight_bounds is None or self.projection.weight_bounds[1] is None:
            self.w_max = 2.
        else:
            self.w_max = self.projection.weight_bounds[1]
        projection.post.register_attribute_history('plateau')
        projection.post.register_attribute_history('backward_activity')
        projection.post.register_attribute_history('backward_dendritic_state')
    
    def reinit(self):
        self.projection.pre.past_ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.past_IS = torch.zeros(self.projection.post.size,
                                                   device=self.projection.post.network.device)
    
    def update(self):
        if not self.projection.pre.past_ET_updated:
            if self.projection.direction in ['forward', 'F']:
                self.projection.pre.past_ET = (torch.clamp(self.projection.pre.forward_activity, 0., 1.) *
                                               self.temporal_discount)
            elif self.projection.direction in ['recurrent', 'R']:
                self.projection.pre.past_ET = (torch.clamp(self.projection.pre.forward_prev_activity, 0., 1.) *
                                               self.temporal_discount)
            self.projection.pre.past_ET_updated = True
        
        if not self.projection.post.past_IS_updated:
            self.projection.post.past_IS = torch.zeros(self.projection.post.size,
                                                       device=self.projection.post.network.device)
            pos_indexes = (self.projection.post.plateau > 0).nonzero().squeeze(1)
            self.projection.post.past_IS[pos_indexes] = (self.projection.post.plateau[pos_indexes] *
                                                         self.temporal_discount)
            self.projection.post.past_IS_updated = True
    
    def step(self):
        # pos error - BTSP weight update
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.forward_activity, 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.forward_prev_activity, 0., 1.)
        
        # pre activity and post plateau for current sample
        delta_weight = IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                             self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))
        
        # pre activity for prev sample and post plateau for current sample
        past_ET = self.projection.pre.past_ET
        delta_weight += IS * ((self.w_max - self.projection.weight) * past_ET.unsqueeze(0) -
                              self.projection.weight * self.dep_ratio * self.q_dep(past_ET).unsqueeze(0))
        
        # pre activity for current sample and post plateau for prev sample
        past_IS = self.projection.post.past_IS.unsqueeze(1)
        delta_weight += past_IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) * self.temporal_discount -
                                   self.projection.weight * self.dep_ratio *
                                   self.q_dep(ET * self.temporal_discount).unsqueeze(0))
        
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = plateau.detach().clone()
        neg_error[plateau > 0.] = 0.
        delta_weight = torch.outer(neg_error, ET)
        self.projection.weight.data += self.learning_rate * delta_weight
    
    @classmethod
    def backward_nudge_activity(cls, layer, store_dynamics=False):
        """
        Update somatic state and activity for all populations that receive a nudge.
        :param layer:
        :param store_dynamics: bool
        """
        for post_pop in layer:
            if post_pop.backward_projections or post_pop is post_pop.network.output_pop:
                if hasattr(post_pop, 'dend_to_soma'):
                    post_pop.prev_activity = post_pop.activity
                    post_pop.activity = post_pop.activation(post_pop.state + post_pop.dend_to_soma)
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
        
        for layer in reversed_layers:
            # update dendritic state variables
            cls.backward_update_layer_dendritic_state(layer)
            for pop in layer:
                for projection in pop:
                    # compute plateau events and nudge somatic state
                    if projection.learning_rule.__class__ == cls:
                        if pop is output_pop:
                            output_pop.dendritic_state = (
                                torch.clamp(target - output_pop.activity, min=-1, max=1))
                            output_pop.plateau = output_pop.dendritic_state.detach().clone()
                            output_pop.dend_to_soma = output_pop.dendritic_state.detach().clone()
                        else:
                            max_units = math.ceil(projection.learning_rule.max_pop_fraction * pop.size)
                            local_loss = pop.dendritic_state.detach().clone()
                            
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
                            
                            pop.plateau[pos_event_indexes] = pop.dendritic_state[pos_event_indexes]
                            pop.plateau[neg_event_indexes] = pop.dendritic_state[neg_event_indexes]
                            pop.dend_to_soma[pos_event_indexes] = pop.dendritic_state[pos_event_indexes]
                            pop.dend_to_soma[neg_event_indexes] = pop.dendritic_state[neg_event_indexes]
                        break
            # update activities
            cls.backward_nudge_activity(layer, store_dynamics=store_dynamics)
        
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
                        pop.past_IS_updated = False
                        projection.pre.past_ET_updated = False
                if store_history:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())


class BTSP_18(LearningRule):
    def __init__(self, projection, neg_rate_th=None, dep_ratio=1., dep_th=0.01, dep_width=0.01, max_pop_fraction=1.,
                 temporal_discount=0.25, stochastic=False, learning_rate=None, relu_gate=True):
        """
        Like the original BTSP class, this method includes both positive and negative modulatory events. In this
        variant, both positive and negative modulatory events nudge the somatic activity.
        Plateaus increment an instructive signal (IS) that decays in two time steps. (Forward) presynaptic activity
        increments an eligibility trace (ET) that decays to zero in two time steps. Positive modulatory events
        result in a BTSP weight update that depends on IS, current weight, and ET.
        Negative modulatory events only occur when postsynaptic activity is above a threshold. Negative modulatory
        events result in a weight update proportional to plateaus and (forward) presynaptic activity.
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
        self.projection.pre.past_ET = torch.zeros(self.projection.pre.size, device=self.projection.pre.network.device)
        self.projection.post.past_IS = torch.zeros(self.projection.post.size,
                                                   device=self.projection.post.network.device)
    
    def update(self):
        if not self.projection.pre.past_ET_updated:
            if self.projection.direction in ['forward', 'F']:
                self.projection.pre.past_ET = (torch.clamp(self.projection.pre.activity, 0., 1.) *
                                               self.temporal_discount)
            elif self.projection.direction in ['recurrent', 'R']:
                self.projection.pre.past_ET = (torch.clamp(self.projection.pre.prev_activity, 0., 1.) *
                                               self.temporal_discount)
            self.projection.pre.past_ET_updated = True
        
        if not self.projection.post.past_IS_updated:
            self.projection.post.past_IS = torch.zeros(self.projection.post.size,
                                                       device=self.projection.post.network.device)
            pos_indexes = (self.projection.post.plateau > 0).nonzero().squeeze(1)
            self.projection.post.past_IS[pos_indexes] = (self.projection.post.plateau[pos_indexes] *
                                                         self.temporal_discount)
            self.projection.post.past_IS_updated = True
    
    def step(self):
        # pos error - BTSP weight update
        plateau = self.projection.post.plateau
        IS = plateau.detach().clone()
        IS[plateau < 0.] = 0.
        IS = IS.unsqueeze(1)
        if self.projection.direction in ['forward', 'F']:
            ET = torch.clamp(self.projection.pre.activity, 0., 1.)
        elif self.projection.direction in ['recurrent', 'R']:
            ET = torch.clamp(self.projection.pre.prev_activity, 0., 1.)
        
        # pre activity and post plateau for current sample
        delta_weight = (IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) -
                              self.projection.weight * self.dep_ratio * self.q_dep(ET).unsqueeze(0))).detach().clone()
        
        # pre activity for prev sample and post plateau for current sample
        past_ET = self.projection.pre.past_ET
        delta_weight += (IS * ((self.w_max - self.projection.weight) * past_ET.unsqueeze(0) -
                               self.projection.weight * self.dep_ratio * self.q_dep(past_ET).unsqueeze(
                    0))).detach().clone()
        
        # pre activity for current sample and post plateau for prev sample
        past_IS = self.projection.post.past_IS.unsqueeze(1)
        delta_weight += (past_IS * ((self.w_max - self.projection.weight) * ET.unsqueeze(0) * self.temporal_discount -
                                    self.projection.weight * self.dep_ratio *
                                    self.q_dep(ET * self.temporal_discount).unsqueeze(0))).detach().clone()
        
        self.projection.weight.data += self.learning_rate * delta_weight
        
        # neg error - weight update proportional to loss and presynaptic activity
        neg_error = plateau.detach().clone()
        neg_error[plateau > 0.] = 0.
        delta_weight = torch.outer(neg_error, ET)
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
                        pop.past_IS_updated = False
                        projection.pre.past_ET_updated = False
                if store_history:
                    if pop.backward_projections or pop is output_pop:
                        if store_dynamics:
                            pop.append_attribute_history('backward_activity', pop.backward_steps_activity)
                        else:
                            pop.append_attribute_history('backward_activity', pop.activity.detach().clone())