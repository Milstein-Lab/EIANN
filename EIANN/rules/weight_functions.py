import torch


def clone_weight(projection, source=None, sign=1, scale=1, source2=None, transpose=False, labels_in_tasks=None):
    """
    Force a projection to exactly copy the weights of another projection (or product of two projections).
    If labels_in_tasks is provided (task-incremental multi-head training), columns from presynaptic units outside the
    current task are then zeroed (see task_column_mask).
    """
    if source is None:
        raise Exception('clone_weight: missing required weight_constraint_kwarg: source')
    network = projection.post.network

    try: 
        source_post_layer, source_post_pop, source_pre_layer, source_pre_pop = source.split('.')
        source_projection = network.layers[source_post_layer].populations[source_post_pop].projections[source_pre_layer][source_pre_pop]
    except:
        source_projection = network.module_dict[source]
    source_weight_data = source_projection.weight.data.clone() * scale * sign

    if source2 is not None:
        source2_post_layer, source2_post_pop, source2_pre_layer, source2_pre_pop = source2.split('.')
        source2_projection = \
            network.layers[source2_post_layer].populations[source2_post_pop].projections[source2_pre_layer][
                source2_pre_pop]
        source2_weight_data = source2_projection.weight.data.clone()
        source_weight_data = source_weight_data * source2_weight_data
    if transpose:
        source_weight_data = source_weight_data.T
    if source_weight_data.shape != projection.weight.data.shape:
        raise Exception('clone_weight: projection shapes do not match; target: %s, %s; source: %s, %s' %
                        (projection.name, str(projection.weight.data.shape), source_projection.name,
                         str(source_weight_data.shape)))
    projection.weight.data = source_weight_data
    if labels_in_tasks is not None:
        apply_task_column_mask(projection, labels_in_tasks)


def normalize_weight(projection, scale, autapses=False, axis=1):
    if not autapses:
        no_autapses(projection)
    weight_sum = torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    valid_rows = torch.nonzero(weight_sum, as_tuple=True)[0]
    projection.weight.data[valid_rows,:] /= weight_sum[valid_rows,:]
    if scale is None:
        if not hasattr(projection.learning_rule, 'weight_norm_scale'):
            projection.learning_rule.weight_norm_scale = weight_sum.detach().clone()
        scale = projection.learning_rule.weight_norm_scale
    projection.weight.data *= scale


def no_autapses(projection):
    if projection.pre is projection.post:
        projection.weight.data.fill_diagonal_(0.)


def receptive_field_mask(projection, receptive_field_size, image_dims=(28, 28), apply_weight_norm=False, **kwargs):
    """
    Enforce receptive fields for a projection by pruning (zeroing) weights outside the receptive field for each unit.
    """
    if len(image_dims) == 2: # Assume there is only 1 channel
        image_dims = [1] + list(image_dims)

    input_size = projection.weight.shape[1]
    expected_size = image_dims[0] * image_dims[1] * image_dims[2]
    if input_size != expected_size:
        print(f"Warning: Projection dimensions ({input_size}) do not match expected dimensions based on receptive field ({expected_size}). Please specify the correct image_dims in the config yaml.")

    if not hasattr(projection, 'weight_mask'):
        projection.weight_mask = _create_receptive_field_mask(n_hidden=projection.weight.shape[0], input_size=input_size, image_dimensions=image_dims, rf_size=receptive_field_size)

    projection.weight.data *= projection.weight_mask

    if apply_weight_norm:
        normalize_weight(projection, **kwargs)



def _create_receptive_field_mask(n_hidden=500, input_size=784, image_dimensions=(1, 28, 28), rf_size=9):
    """
    Create a mask for hidden units with randomly positioned receptive fields.
    
    Args:
        n_hidden: Number of hidden units (500)
        input_size: Input dimension (784 for MNIST, 3072 for CIFAR-10)
        image_dimensions: Tuple of (channels, height, width) - (1, 28, 28) for MNIST; (3, 32, 32) for CIFAR-10
        rf_size: Receptive field size (default = 9x9)

    Returns:
        torch.Tensor: Binary mask of shape [n_hidden, input_size]
    """
    channels, img_height, img_width = image_dimensions
    
    # Initialize mask with zeros
    mask = torch.zeros(n_hidden, input_size)
    
    # Calculate the maximum starting positions for receptive fields
    max_rf_start_row = img_height - rf_size + 1  # 28 - 9 + 1 = 20 (MNIST)
    max_rf_start_col = img_width - rf_size + 1   # 28 - 9 + 1 = 20 (MNIST)
    
    # Create receptive fields for each hidden unit
    for unit_idx in range(n_hidden):
        # Randomly sample the top-left corner position of the receptive field
        rf_start_row = torch.randint(0, max_rf_start_row, (1,)).item()
        rf_start_col = torch.randint(0, max_rf_start_col, (1,)).item()
        
        # Get a view of this unit's mask reshaped to (channels, height, width)
        rf_mask_3d = mask[unit_idx].view(channels, img_height, img_width)
        
        # Set the receptive field region to 1 across all channels
        rf_mask_3d[:, rf_start_row:rf_start_row + rf_size, 
                   rf_start_col:rf_start_col + rf_size] = 1
    
    return mask

# ---------------------------------------------------------------------------------------------------------------------
# Task-incremental multi-head structure. Output-layer populations are split into one "head" per task: output E units by
# the classes in each task, and other output-layer populations (e.g. inhibitory interneurons) into contiguous blocks.
# ---------------------------------------------------------------------------------------------------------------------
def get_unit_tasks(population, labels_in_tasks):
    """
    Task index of each unit of an output-layer population.
    :param population: :class:'Population'
    :param labels_in_tasks: list of lists of int; the classes in each task
    :return: tensor of int (population.size,)
    """
    if population is population.network.output_pop:
        unit_tasks = torch.empty(population.size, dtype=torch.long)
        for task, labels in enumerate(labels_in_tasks):
            unit_tasks[list(labels)] = task
        return unit_tasks
    return torch.arange(population.size) * len(labels_in_tasks) // population.size


def get_current_task(network, num_tasks):
    """
    Current task index, read from the task counter of the network's continual-learning rules (which
    network.update_CL_states() advances between tasks).
    """
    from .base_classes import ContinualLearningMixin
    for projection in network.projections.values():
        if isinstance(projection.learning_rule, ContinualLearningMixin):
            return min(projection.learning_rule.task_num, num_tasks - 1)
    return 0


def apply_base_constraint(projection, base_constraint, base_constraint_kwargs):
    if base_constraint is None:
        return
    if isinstance(base_constraint, str):
        import EIANN.external as external
        base_constraint = globals().get(base_constraint) or getattr(external, base_constraint)
    base_constraint(projection, **(base_constraint_kwargs or {}))


def get_inactive_task_columns(projection, labels_in_tasks):
    """
    Boolean mask of the weight columns whose presynaptic (output-layer) unit is outside the current task.
    """
    pre_tasks = get_unit_tasks(projection.pre, labels_in_tasks).to(projection.weight.device)
    current_task = get_current_task(projection.post.network, len(labels_in_tasks))
    return pre_tasks != current_task


def apply_task_column_mask(projection, labels_in_tasks):
    projection.weight.data[:, get_inactive_task_columns(projection, labels_in_tasks)] = 0.


def task_block_mask(projection, labels_in_tasks, base_constraint=None, base_constraint_kwargs=None):
    """
    Weight constraint for projections between output-layer populations in task-incremental multi-head networks: only
    connections within the same task's head are kept, so each head is an independent E/I subnetwork on the shared hidden
    layers. Any original constraint of the projection (base_constraint) is applied first.
    """
    apply_base_constraint(projection, base_constraint, base_constraint_kwargs)
    post_tasks = get_unit_tasks(projection.post, labels_in_tasks)
    pre_tasks = get_unit_tasks(projection.pre, labels_in_tasks)
    mask = (post_tasks.unsqueeze(1) == pre_tasks.unsqueeze(0)).to(projection.weight.device)
    projection.weight.data *= mask


def task_column_mask(projection, labels_in_tasks, base_constraint=None, base_constraint_kwargs=None):
    """
    Weight constraint for top-down projections from the output layer in task-incremental multi-head networks: columns
    from output-layer units outside the current task are zeroed, so only the active head sends top-down signals to the
    hidden layers while training. Any original constraint of the projection (base_constraint) is applied first.
    Projections that use clone_weight get the same mask through clone_weight(labels_in_tasks=...).
    The values of the masked columns are stashed on the projection (task_column_mask_stash) and restored at the next
    call, so a head's top-down weights are kept while it is inactive and come back when its task starts (zeroing them in
    place would lose them for good on projections that are fixed or not re-cloned). Masked columns are frozen at their
    stashed values: learning-rule updates to them are discarded. The base constraint sees the full (restored) weights.
    """
    stash = getattr(projection, 'task_column_mask_stash', None)
    if stash is not None:
        inactive_columns, inactive_weight = stash
        projection.weight.data[:, inactive_columns] = inactive_weight
    apply_base_constraint(projection, base_constraint, base_constraint_kwargs)
    inactive_columns = get_inactive_task_columns(projection, labels_in_tasks)
    projection.task_column_mask_stash = (inactive_columns, projection.weight.data[:, inactive_columns].clone())
    projection.weight.data[:, inactive_columns] = 0.
