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


class Backprop_EWC_LR(Backprop_CL):
    """
    Elastic weight consolidation (Kirkpatrick et al., 2017, arXiv:1612.00796) implemented as a learning rate
    modulation instead of a loss penalty. During each task, a uniform random subset (reservoir sample) of the training
    samples is cached. At the end of each task (update_CL_states), the diagonal of the empirical Fisher information
    (squared per-sample gradient of the task loss, averaged over the cached samples) is computed at the current weights
    and added to the Fisher summed over all previous tasks. The summed (raw) Fisher is passed through a piecewise
    linear function that decreases from 1 (Fisher <= silr_threshold) to silr_min (Fisher >= silr_threshold +
    silr_width), and during subsequent tasks the gradient of the task loss for each weight is scaled by that value:
        grad_p = lr_mod_func(sum_t F_pt) * d task_loss(output, target) / d W_p
    No penalty is added to the loss. Scaling the gradient is equivalent to scaling the per-weight learning rate only
    for plain SGD (no momentum or weight decay). See Backprop_CL for task_incremental and task_classes.
    """
    
    def __init__(self, projection, fisher_num_samples=1000, silr_threshold=1e-4, silr_width=1e-4, silr_min=0.,
                 task_incremental=False, task_classes=None, learning_rate=None):
        super().__init__(projection, task_incremental=task_incremental, task_classes=task_classes,
                         learning_rate=learning_rate)
        self.fisher_num_samples = int(fisher_num_samples)
        self.fisher_sum = None
        self.num_fisher_tasks = 0
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
    
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
    
    def update_CL_states(self):
        # The first projection called at the end of a task computes the Fisher for all projections using this rule
        if self.num_fisher_tasks == self.task_num:
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
                learning_rule = projection.learning_rule
                if learning_rule.fisher_sum is None:
                    learning_rule.fisher_sum = fisher
                else:
                    learning_rule.fisher_sum = learning_rule.fisher_sum + fisher
                learning_rule.lr_mod = learning_rule.lr_mod_func(learning_rule.fisher_sum)
                learning_rule.num_fisher_tasks += 1
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
        network.optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for projection in ewc_projections:
                if projection.learning_rule.task_num > 0 and projection.weight.grad is not None:
                    projection.weight.grad *= projection.learning_rule.lr_mod
        
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


class Backprop_SI_LR(Backprop_CL):
    """
    Synaptic Intelligence (Zenke et al., 2017, arXiv:1703.04200) implemented as a learning rate modulation instead of a
    loss penalty. The per-weight importance omega is computed as in Backprop_SI: during each task the path integral
    small_omega += -g * delta_weight is accumulated, where g is the gradient of the task loss (before modulation) and
    delta_weight is the actual weight change of each train step (after the optimizer step and weight clamping), and at
    the end of each task (update_CL_states) omega = relu(omega + small_omega / (total task weight change ** 2 + si_xi)).
    omega is passed through a piecewise linear function that decreases from 1 (omega <= silr_threshold) to silr_min
    (omega >= silr_threshold + silr_width), and during subsequent tasks the gradient of the task loss for each weight is
    scaled by that value:
        grad_p = lr_mod_func(omega_p) * d task_loss(output, target) / d W_p
    No penalty is added to the loss. Weights frozen by lr_mod = 0 gain no further importance, since their delta_weight
    is 0. Scaling the gradient is equivalent to scaling the per-weight learning rate only for plain SGD (no momentum or
    weight decay). See Backprop_CL for task_incremental and task_classes.
    """
    
    def __init__(self, projection, si_xi=1e-3, silr_threshold=1., silr_width=1., silr_min=0., task_incremental=False,
                 task_classes=None, learning_rate=None):
        super().__init__(projection, task_incremental=task_incremental, task_classes=task_classes,
                         learning_rate=learning_rate)
        self.si_xi = si_xi
        self.omega = None
        self.small_omega = torch.zeros_like(projection.weight.data)
        # Weights are initialized by the network after the projection is built, so this is set on the first backward
        self.task_start_weight = None
        self.unreg_grad = None
        self.prev_weight = None
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(1., silr_min, silr_threshold, silr_threshold + silr_width)
    
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
            # Clamp at zero, as in the reference implementation (ganguli-lab/pathint)
            self.omega = torch.relu(omega)
            self.lr_mod = self.lr_mod_func(self.omega)
        self.task_start_weight = weight
        self.small_omega = torch.zeros_like(weight)
        super().update_CL_states()
    
    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):

        si_projections = cls.get_projections(network)

        network.optimizer.zero_grad()
        loss = si_projections[0].learning_rule.task_loss(network, output, target)
        loss.backward()

        with torch.no_grad():
            for projection in si_projections:
                learning_rule = projection.learning_rule
                if learning_rule.task_start_weight is None:
                    learning_rule.task_start_weight = projection.weight.detach().clone()
                if projection.weight.grad is None:
                    learning_rule.unreg_grad = torch.zeros_like(projection.weight.data)
                else:
                    learning_rule.unreg_grad = projection.weight.grad.detach().clone()
                    if learning_rule.task_num > 0:
                        projection.weight.grad *= learning_rule.lr_mod
                learning_rule.prev_weight = projection.weight.detach().clone()

        network.optimizer.step()


class Backprop_SI_ELR(Backprop_CL):
    """
    Synaptic Intelligence (Zenke et al., 2017, arXiv:1703.04200) with an elastic learning rate (as in BTSP_ELR_CL)
    instead of a loss penalty. The per-weight importance omega and the anchor weight (the weight at the end of the
    previous task) are computed as in Backprop_SI_LR. During subsequent tasks the gradient of the task loss for each
    weight is scaled by lr_mod:
        grad_p = lr_mod_p * d task_loss(output, target) / d W_p
    and after every train step (after the optimizer step and weight clamping), lr_mod is updated continuously:
        lr_mod_p += (d task_loss / d W_p != 0) * omega_p * lr_mod_func(|W_p - W_anchor_p|)
        lr_mod_p = clamp(lr_mod_p, elr_min, 1)
    where lr_mod_func is a piecewise linear function that increases from -elr_scale (distance <= elr_threshold) to
    elr_scale (distance >= elr_threshold + elr_width). Important weights near their anchor lose learning rate, and
    important weights far from it recover learning rate. lr_mod starts at 1, stays at 1 during the first task (omega is
    not defined before the first task switch), and is not reset at task boundaries. No penalty is added to the loss.
    Scaling the gradient is equivalent to scaling the per-weight learning rate only for plain SGD (no momentum or
    weight decay). See Backprop_CL for task_incremental and task_classes.
    """
    # lr_mod is updated during training, not set at the task switch (used by scripts/get_cl_metrics.py)
    continuous_lr_mod = True

    def __init__(self, projection, si_xi=1e-3, elr_threshold=0., elr_width=0.5, elr_scale=0.1, elr_min=0.01,
                 task_incremental=False, task_classes=None, learning_rate=None):
        super().__init__(projection, task_incremental=task_incremental, task_classes=task_classes,
                         learning_rate=learning_rate)
        self.si_xi = si_xi
        self.omega = None
        self.small_omega = torch.zeros_like(projection.weight.data)
        # Weights are initialized by the network after the projection is built, so this is set on the first backward
        self.task_start_weight = None
        self.unreg_grad = None
        self.prev_weight = None
        self.elr_threshold = elr_threshold
        self.elr_width = elr_width
        self.elr_scale = elr_scale
        self.elr_min = elr_min
        self.lr_mod = torch.ones_like(projection.weight.data)
        self.lr_mod_func = pwlin(-self.elr_scale, self.elr_scale, self.elr_threshold,
                                 self.elr_threshold + self.elr_width)

    def update(self):
        # Called after optimizer.step() and constrain_weights_and_biases(), so delta_weight includes weight clamping
        if self.prev_weight is not None:
            weight = self.projection.weight.detach()
            delta_weight = weight - self.prev_weight
            self.small_omega -= self.unreg_grad * delta_weight
            self.prev_weight = None

            # elastic learning rate modulation
            if self.omega is not None:
                distance_to_anchor = torch.abs(weight - self.task_start_weight)
                gate = (self.unreg_grad != 0.).float()
                self.lr_mod += gate * self.omega * self.lr_mod_func(distance_to_anchor)
                self.lr_mod.clamp_(min=self.elr_min, max=1.)

    def update_CL_states(self):
        weight = self.projection.weight.detach().clone()
        if self.task_start_weight is not None:
            omega = self.small_omega / ((weight - self.task_start_weight) ** 2 + self.si_xi)
            if self.omega is not None:
                omega = self.omega + omega
            # Clamp at zero, as in the reference implementation (ganguli-lab/pathint)
            self.omega = torch.relu(omega)
        self.task_start_weight = weight
        self.small_omega = torch.zeros_like(weight)
        super().update_CL_states()

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):

        si_projections = cls.get_projections(network)

        network.optimizer.zero_grad()
        loss = si_projections[0].learning_rule.task_loss(network, output, target)
        loss.backward()

        with torch.no_grad():
            for projection in si_projections:
                learning_rule = projection.learning_rule
                if learning_rule.task_start_weight is None:
                    learning_rule.task_start_weight = projection.weight.detach().clone()
                if projection.weight.grad is None:
                    learning_rule.unreg_grad = torch.zeros_like(projection.weight.data)
                else:
                    learning_rule.unreg_grad = projection.weight.grad.detach().clone()
                    if learning_rule.task_num > 0:
                        projection.weight.grad *= learning_rule.lr_mod
                learning_rule.prev_weight = projection.weight.detach().clone()

        network.optimizer.step()


class Backprop_SI_HD(Backprop_CL):
    """
    Synaptic Intelligence (Zenke et al., 2017, arXiv:1703.04200) with a per-weight learning rate adapted by
    hypergradient descent (Baydin et al., 2017, arXiv:1703.04782; Franceschi et al., 2017, arXiv:1703.01785) on the
    SI-regularized loss, instead of a loss penalty. The per-weight importance omega and the anchor weight (the weight at
    the end of the previous task) are computed as in Backprop_SI_LR. Weights are updated with the gradient g of the
    unregularized task loss, scaled by a per-weight learning rate lr = learning_rate * lr_mod:
        W_p[t+1] = W_p[t] - lr_p[t] * g_p[t]
    so dW_p[t] / dlr_p[t-1] = -g_p[t-1], and before every train step lr descends the hypergradient of the regularized
    loss task_loss + hd_beta * sum_p (omega_p * (W_p - W_anchor_p) ** 2).sum():
        lr_p[t] = lr_p[t-1] + hd_gamma * g_p[t-1] * (g_p[t] + 2 * hd_beta * omega_p * (W_p[t] - W_anchor_p))
        lr_p[t] = clamp(lr_p[t], lr_min, lr_max)
    lr starts at learning_rate. During the first task omega is not defined, so this is plain per-weight hypergradient
    descent. lr is not reset at task boundaries. No penalty is added to the loss. The hypergradient ignores weight
    clamping (it uses -g_p[t-1] even when the weight was clamped). Scaling the gradient is equivalent to scaling the
    per-weight learning rate only for plain SGD (no momentum or weight decay). See Backprop_CL for task_incremental and
    task_classes.
    """
    # lr_mod is updated during training, not set at the task switch (used by scripts/get_cl_metrics.py)
    continuous_lr_mod = True

    def __init__(self, projection, hd_gamma=1e-3, hd_beta=1., si_xi=1e-3, lr_min=0., lr_max=None,
                 task_incremental=False, task_classes=None, learning_rate=None):
        super().__init__(projection, task_incremental=task_incremental, task_classes=task_classes,
                         learning_rate=learning_rate)
        self.hd_gamma = hd_gamma
        self.hd_beta = hd_beta
        self.si_xi = si_xi
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.omega = None
        self.small_omega = torch.zeros_like(projection.weight.data)
        # Weights are initialized by the network after the projection is built, so this is set on the first backward
        self.task_start_weight = None
        self.unreg_grad = None
        self.prev_grad = None
        self.prev_weight = None
        self.lr_mod = torch.ones_like(projection.weight.data)

    def update_lr(self, grad):
        """
        Hypergradient step on the per-weight learning rate, using the task loss gradient of the previous and the
        current train step.
        :param grad: tensor; gradient of the task loss at the current weights
        """
        if self.prev_grad is None:
            return
        hypergrad = grad
        if self.omega is not None:
            hypergrad = hypergrad + 2. * self.hd_beta * self.omega * (self.projection.weight.detach() -
                                                                      self.task_start_weight)
        lr = self.learning_rate * self.lr_mod + self.hd_gamma * self.prev_grad * hypergrad
        lr.clamp_(min=self.lr_min, max=self.lr_max)
        self.lr_mod = lr / self.learning_rate

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
            # Clamp at zero, as in the reference implementation (ganguli-lab/pathint)
            self.omega = torch.relu(omega)
        self.task_start_weight = weight
        self.small_omega = torch.zeros_like(weight)
        super().update_CL_states()

    @classmethod
    def backward(cls, network, output, target, store_history=False, store_dynamics=False):

        si_projections = cls.get_projections(network)

        network.optimizer.zero_grad()
        loss = si_projections[0].learning_rule.task_loss(network, output, target)
        loss.backward()

        with torch.no_grad():
            for projection in si_projections:
                learning_rule = projection.learning_rule
                if learning_rule.task_start_weight is None:
                    learning_rule.task_start_weight = projection.weight.detach().clone()
                if projection.weight.grad is None:
                    learning_rule.unreg_grad = torch.zeros_like(projection.weight.data)
                else:
                    learning_rule.unreg_grad = projection.weight.grad.detach().clone()
                learning_rule.update_lr(learning_rule.unreg_grad)
                if projection.weight.grad is not None:
                    projection.weight.grad *= learning_rule.lr_mod
                learning_rule.prev_grad = learning_rule.unreg_grad
                learning_rule.prev_weight = projection.weight.detach().clone()

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
