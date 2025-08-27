import torch


def clone_weight(projection, source=None, sign=1, scale=1, source2=None, transpose=False):
    """
    Force a projection to exactly copy the weights of another projection (or product of two projections).
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


def normalize_weight(projection, scale, autapses=False, axis=1):
    if not autapses:
        no_autapses(projection)
    weight_sum = torch.sum(torch.abs(projection.weight.data), axis=axis).unsqueeze(1)
    valid_rows = torch.nonzero(weight_sum, as_tuple=True)[0]
    projection.weight.data[valid_rows,:] /= weight_sum[valid_rows,:]
    projection.weight.data *= scale


def no_autapses(projection):
    if projection.pre is projection.post:
        projection.weight.data.fill_diagonal_(0.)


def receptive_field_mask(projection, receptive_field_size, image_dims=[1, 28, 28], apply_weight_norm=False, **kwargs):
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