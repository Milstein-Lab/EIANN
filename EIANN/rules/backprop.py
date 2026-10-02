import torch
from .base_classes import LearningRule, BiasLearningRule, ContinualLearningMixin
from EIANN.utils import pwlin


class Backprop(LearningRule):
    def __init__(self, projection, learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()


class Backprop_INEL_A(LearningRule):
    # Target weight is the unit mean
    def __init__(self, projection, inel_threshold=0.5, learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.inel_threshold = inel_threshold
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    if projection.learning_rule.task_num > 0:
                        unit_mean_weights = torch.mean(projection.weight.data, dim=1)
                        inel_indexes = (torch.abs(projection.weight.data - unit_mean_weights.unsqueeze(1)) <
                                        projection.learning_rule.inel_threshold).nonzero(as_tuple=True)
                        projection.weight.grad[inel_indexes] = 0.
        
        network.optimizer.step()


class Backprop_INEL_B(LearningRule):
    # Target weight is the projection mean
    def __init__(self, projection, inel_threshold=0.5, learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.inel_threshold = inel_threshold
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    if projection.learning_rule.task_num > 0:
                        projection_mean_weight = torch.mean(projection.weight.data)
                        inel_indexes = (torch.abs(projection.weight.data - projection_mean_weight) <
                                        projection.learning_rule.inel_threshold).nonzero(as_tuple=True)
                        projection.weight.grad[inel_indexes] = 0.
        
        network.optimizer.step()


class Backprop_SILR_A(LearningRule):
    """
    Inspired by "Synaptic Intelligence" (Zenke et al., 2019). Parameter updates are accumulated during a task,
    and learning rates are modulated during subsequent tasks.
    """
    def __init__(self, projection, silr_threshold=0.5, silr_width=0.5, silr_min=0., learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
        self.accum_delta_weight = torch.zeros_like(projection.weight.data)
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
        self.lr_mod = self.lr_mod_func(torch.abs(self.accum_delta_weight))
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    delta_weight = (-projection.learning_rule.learning_rate * projection.learning_rule.lr_mod *
                                    projection.weight.grad)
                    projection.learning_rule.accum_delta_weight += delta_weight
                    if projection.learning_rule.task_num > 0:
                        projection.weight.grad *= projection.learning_rule.lr_mod
        
        network.optimizer.step()


class Backprop_SILR_B(LearningRule):
    """
    Inspired by "Synaptic Intelligence" (Zenke et al., 2019). Parameter updates are accumulated during a task,
    and learning rates are modulated during subsequent tasks. Also depends on distance of each weight from its value in
    the previous task.
    """
    
    def __init__(self, projection, silr_threshold=0.5, silr_width=0.5, silr_min=0., learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
        self.accum_delta_weight = torch.zeros_like(projection.weight.data)
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
        self.prev_task_accum_delta_weight = self.accum_delta_weight.detach().clone()
        self.prev_task_weights = self.projection.weight.data.detach().clone()
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    if projection.learning_rule.task_num > 0:
                        lr_mod = projection.learning_rule.lr_mod_func(
                            torch.abs(projection.learning_rule.prev_task_accum_delta_weight *
                                      (projection.weight.data - projection.learning_rule.prev_task_weights)))
                        projection.weight.grad *= lr_mod
                    else:
                        lr_mod = 1.
                    delta_weight = (-projection.learning_rule.learning_rate * lr_mod *
                                    projection.weight.grad)
                    projection.learning_rule.accum_delta_weight += delta_weight
        
        network.optimizer.step()


class Backprop_SILR_C(LearningRule):
    """
    Inspired by "Synaptic Intelligence" (Zenke et al., 2019). The product of gradients and parameter updates are
    accumulated during a task, and learning rates are modulated during subsequent tasks.
    """
    
    def __init__(self, projection, silr_threshold=0.5, silr_width=0.5, silr_min=0., learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
        self.accum_delta_weight = torch.zeros_like(projection.weight.data)
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
        self.lr_mod = self.lr_mod_func(self.accum_delta_weight)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    delta_weight = (-projection.learning_rule.learning_rate * projection.learning_rule.lr_mod *
                                    projection.weight.grad)
                    projection.learning_rule.accum_delta_weight -= projection.weight.grad * delta_weight
                    if projection.learning_rule.task_num > 0:
                        projection.weight.grad *= projection.learning_rule.lr_mod
        
        network.optimizer.step()


class Backprop_SILR_D(LearningRule):
    """
    Inspired by "Synaptic Intelligence" (Zenke et al., 2019). The product of gradients and parameter updates are
    accumulated during a task, and learning rates are modulated during subsequent tasks. Also depends on distance of
    each weight from its value in the previous task.
    """
    
    def __init__(self, projection, silr_threshold=0.5, silr_width=0.5, silr_min=0., learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
        self.accum_delta_weight = torch.zeros_like(projection.weight.data)
        self.task_num = 0
    
    def update_CL_states(self):
        self.task_num += 1
        self.prev_task_accum_delta_weight = self.accum_delta_weight.detach().clone()
        self.prev_task_weights = self.projection.weight.data.detach().clone()
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        loss = network.criterion(output, target)
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in network.projections.values():
                if projection.learning_rule.__class__ == cls:
                    if projection.learning_rule.task_num > 0:
                        lr_mod = projection.learning_rule.lr_mod_func(
                            torch.abs(projection.learning_rule.prev_task_accum_delta_weight *
                                      (projection.weight.data - projection.learning_rule.prev_task_weights)))
                        projection.weight.grad *= lr_mod
                    else:
                        lr_mod = 1.
                    delta_weight = (-projection.learning_rule.learning_rate * lr_mod *
                                    projection.weight.grad)
                    projection.learning_rule.accum_delta_weight -= projection.weight.grad * delta_weight
        
        network.optimizer.step()


class Backprop_CL(ContinualLearningMixin, LearningRule):
    """
    Backprop for continual learning. Counts tasks (update_CL_states is called by the network between tasks) and, when
    task_incremental is True, computes the loss only on the output units of the current task's classes
    (task_classes[task_num]). With task_incremental=False, it is equivalent to Backprop. Base class for the CL rules
    below. All learned projections in the network should use the same rule, since any other backward method would also
    step the shared optimizer.
    """

    def __init__(self, projection, task_incremental=False, task_classes=None, learning_rate=None):
        super().__init__(projection, learning_rate)
        projection.weight.requires_grad = True
        self.init_continual_learning(task_incremental, task_classes)

    @classmethod
    def get_projections(cls, network):
        return [projection for projection in network.projections.values()
                if projection.learning_rule.__class__ == cls]

    def task_loss(self, network, output, target, task_num=None):
        if not self.task_incremental:
            return network.criterion(output, target)
        task_idx = self.get_task_classes(task_num)
        return network.criterion(output[..., task_idx], target[..., task_idx])

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        projections = cls.get_projections(network)
        loss = projections[0].learning_rule.task_loss(network, output, target)
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()


class Backprop_EWC(Backprop_CL):
    """
    Elastic weight consolidation (Kirkpatrick et al., 2017, arXiv:1612.00796). During each task, a uniform random
    subset (reservoir sample) of the training samples is cached. At the end of each task (update_CL_states), the
    diagonal of the empirical Fisher information (squared per-sample gradient of the task loss, averaged over the
    cached samples) is computed at the current weights, and the weights are stored as that task's anchor. The loss
    during subsequent tasks is:
        task_loss(output, target) + sum_p ewc_lambda_p * sum_t (F_pt * (W_p - W*_pt) ** 2).sum()
    See Backprop_CL for task_incremental and task_classes.
    """
    
    def __init__(self, projection, ewc_lambda=1., fisher_num_samples=1000, task_incremental=False, task_classes=None,
                 learning_rate=None):
        super().__init__(projection, task_incremental=task_incremental, task_classes=task_classes,
                         learning_rate=learning_rate)
        self.ewc_lambda = ewc_lambda
        self.fisher_num_samples = int(fisher_num_samples)
        self.fisher_list = []
        self.anchor_weight_list = []
    
    @classmethod
    def get_buffer(cls, network):
        """
        Samples cached from the current task, shared by all projections using this rule.
        """
        if not hasattr(network, 'ewc_buffer'):
            generator = torch.Generator()
            generator.manual_seed(network.seed if network.seed is not None else 0)
            # Store the generator state rather than the generator, since torch.Generator cannot be pickled
            network.ewc_buffer = {'data': [], 'target': [], 'num_seen': 0, 'rng_state': generator.get_state()}
        return network.ewc_buffer
    
    @classmethod
    def cache_sample(cls, network, data, target, max_samples):
        """
        Reservoir sampling: after n samples, each has been kept with equal probability max_samples / n.
        """
        buffer = cls.get_buffer(network)
        data = data.detach().clone()
        target = target.detach().clone()
        if buffer['num_seen'] < max_samples:
            buffer['data'].append(data)
            buffer['target'].append(target)
        else:
            generator = torch.Generator()
            generator.set_state(buffer['rng_state'])
            idx = torch.randint(0, buffer['num_seen'] + 1, (1,), generator=generator).item()
            buffer['rng_state'] = generator.get_state()
            if idx < max_samples:
                buffer['data'][idx] = data
                buffer['target'][idx] = target
        buffer['num_seen'] += 1
    
    def ewc_penalty(self):
        penalty = 0.
        for fisher, anchor_weight in zip(self.fisher_list, self.anchor_weight_list):
            penalty = penalty + (fisher * (self.projection.weight - anchor_weight) ** 2).sum()
        return penalty
    
    def update_CL_states(self):
        # The first projection called at the end of a task computes the Fisher for all projections using this rule
        if len(self.fisher_list) == self.task_num:
            network = self.projection.post.network
            ewc_projections = self.get_projections(network)
            buffer = self.get_buffer(network)
            weights = [projection.weight for projection in ewc_projections]
            fishers = [torch.zeros_like(weight) for weight in weights]
            num_samples = len(buffer['data'])
            for data, target in zip(buffer['data'], buffer['target']):
                output = network.forward(data)
                loss = self.task_loss(network, output, target, task_num=self.task_num)
                grads = torch.autograd.grad(loss, weights, allow_unused=True)
                for fisher, grad in zip(fishers, grads):
                    if grad is not None:
                        fisher += grad.detach() ** 2
            for projection, fisher in zip(ewc_projections, fishers):
                if num_samples > 0:
                    fisher /= num_samples
                projection.learning_rule.fisher_list.append(fisher)
                projection.learning_rule.anchor_weight_list.append(projection.weight.detach().clone())
            buffer['data'] = []
            buffer['target'] = []
            buffer['num_seen'] = 0
        super().update_CL_states()
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        ewc_projections = cls.get_projections(network)
        first_rule = ewc_projections[0].learning_rule
        cls.cache_sample(network, network.input_pop.activity, target, first_rule.fisher_num_samples)
        
        loss = first_rule.task_loss(network, output, target)
        for projection in ewc_projections:
            if projection.learning_rule.task_num > 0:
                loss = loss + projection.learning_rule.ewc_lambda * projection.learning_rule.ewc_penalty()
        
        network.optimizer.zero_grad()
        loss.backward()
        network.optimizer.step()


class Backprop_SI(Backprop_CL):
    """
    Synaptic Intelligence (Zenke et al., 2017, arXiv:1703.04200). During each task, the per-weight importance is
    accumulated online as the path integral small_omega += -g * delta_weight, where g is the gradient of the
    unregularized task loss and delta_weight is the actual weight change of each train step (after the optimizer step
    and weight clamping). At the end of each task (update_CL_states), the cumulative importance
    omega = relu(omega + small_omega / (total task weight change ** 2 + si_xi)), the anchor weight is set to the current
    weight, and small_omega is reset. The loss during subsequent tasks is:
        task_loss(output, target) + sum_p si_lambda_p * (omega_p * (W_p - W_anchor_p) ** 2).sum()
    See Backprop_CL for task_incremental and task_classes.
    """
    
    def __init__(self, projection, si_lambda=1., si_xi=1e-3, task_incremental=False, task_classes=None,
                 learning_rate=None):
        super().__init__(projection, task_incremental=task_incremental, task_classes=task_classes,
                         learning_rate=learning_rate)
        self.si_lambda = si_lambda
        self.si_xi = si_xi
        self.omega = None
        self.anchor_weight = None
        self.small_omega = torch.zeros_like(projection.weight.data)
        # Weights are initialized by the network after the projection is built, so this is set on the first backward
        self.task_start_weight = None
        self.unreg_grad = None
        self.prev_weight = None
    
    def si_penalty(self):
        if self.omega is None:
            return 0.
        return (self.omega * (self.projection.weight - self.anchor_weight) ** 2).sum()
    
    def update(self):
        # Called after optimizer.step() and constrain_weights_and_biases(), so delta_weight includes weight clamping
        if self.prev_weight is not None:
            delta_weight = self.projection.weight.detach() - self.prev_weight
            self.small_omega -= self.unreg_grad * delta_weight
            self.prev_weight = None
    
    def update_CL_states(self):
        weight = self.projection.weight.detach().clone()
        if self.task_start_weight is not None:
            omega = self.small_omega / ((weight - self.task_start_weight) ** 2 + self.si_xi)
            if self.omega is not None:
                omega = self.omega + omega
            # Clamp at zero, as in the reference implementation (ganguli-lab/pathint), so the penalty never pushes
            # weights away from their anchor
            self.omega = torch.relu(omega)
        self.anchor_weight = weight
        self.task_start_weight = weight.clone()
        self.small_omega = torch.zeros_like(weight)
        super().update_CL_states()
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        
        si_projections = cls.get_projections(network)
        
        network.optimizer.zero_grad()
        loss = si_projections[0].learning_rule.task_loss(network, output, target)
        loss.backward()
        
        penalty = 0.
        for projection in si_projections:
            learning_rule = projection.learning_rule
            if learning_rule.task_start_weight is None:
                learning_rule.task_start_weight = projection.weight.detach().clone()
            if projection.weight.grad is None:
                learning_rule.unreg_grad = torch.zeros_like(projection.weight.data)
            else:
                learning_rule.unreg_grad = projection.weight.grad.detach().clone()
            learning_rule.prev_weight = projection.weight.detach().clone()
            if learning_rule.task_num > 0:
                penalty = penalty + learning_rule.si_lambda * learning_rule.si_penalty()
        
        if torch.is_tensor(penalty):
            penalty.backward()
        network.optimizer.step()


class BackpropBias(BiasLearningRule):
    def __init__(self, population, learning_rate=None):
        super().__init__(population, learning_rate)
        population.bias.requires_grad = True
    
    backward = Backprop.backward


class Backprop_DendriticLoss(LearningRule):
    def __init__(self, projection, source, learning_rate=None):
        """

        :param projection: :class:'nn.Linear'
        :param source: str ('layer_name.pop_name')
        :param learning_rate: float
        """
        super().__init__(projection, learning_rate)
        source_post_layer, source_post_pop = source.split('.')
        self.source_pop = projection.post.network.layers[source_post_layer].populations[source_post_pop]
        projection.weight.requires_grad = True
        
        # Create one optimizer the source and register the projection parameters
        if hasattr(self.source_pop, 'local_optimizer'):
            self.source_pop.local_optimizer.add_param_group({'params': projection.parameters(),
                                                             'lr': self.learning_rate})
        else:
            self.source_pop.local_optimizer = torch.optim.SGD(projection.parameters(), lr=self.learning_rate)
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):
        """

        :param network:
        :param output:
        :param target:
        :param store_history:
        :param store_dynamics:
        """
        local_optimizer_list = []
        
        reversed_layers = list(network)[1:]
        reversed_layers.reverse()
        
        for layer in reversed_layers:
            for pop in layer:
                for projection in pop:
                    if projection.learning_rule.__class__ == cls:
                        source_pop = projection.learning_rule.source_pop
                        local_optimizer = source_pop.local_optimizer
                        if local_optimizer not in local_optimizer_list:
                            local_optimizer_list.append(local_optimizer)
                            local_target = torch.zeros(source_pop.size, device=network.device)
                            local_loss = network.criterion(source_pop.forward_dendritic_state, local_target)
                            local_optimizer.zero_grad()
                            local_loss.backward()
                            local_optimizer.step()
