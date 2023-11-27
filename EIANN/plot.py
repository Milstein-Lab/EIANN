import torch
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from skimage import metrics
import scipy.stats as stats

from tqdm.autonotebook import tqdm
from copy import copy

from . import utils


def update_plot_defaults():
    plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'axes.linewidth':1.2,
                     'xtick.major.size': 6,
                     'xtick.major.width': 1.2,
                     'ytick.major.size': 6,
                     'ytick.major.width': 1.2,
                     'legend.frameon': False,
                     'legend.handletextpad': 0.1,
                     'figure.figsize': [10.0, 3.0],
                     'svg.fonttype': 'none',
                     'text.usetex': False})


def clean_axes(axes, left=True, right=False):
    """
    Remove top and right axes from pyplot axes object.
    :param axes: list of pyplot.Axes
    :param top: bool
    :param left: bool
    :param right: bool
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        if not right:
            axis.spines['right'].set_visible(False)
        if not left:
            axis.spines['left'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()


# *******************************************************************
# Network summary functions
# *******************************************************************
def plot_EIANN_1_hidden_autoenc_config_summary(network, test_dataloader, sorted_output_idx=None, title=None):
    """

    :param network:
    :param test_dataloader:
    :param sorted_output_idx: tensor of int
    :param title: str
    """
    assert len(test_dataloader) == 1, 'Dataloader must have a single large batch'

    network.test(test_dataloader, store_dynamics=True, status_bar=True)

    if title is None:
        title_str = ''
    else:
        title_str = '%s ' % title

    max_rows = 1
    cols = len(network.layers) - 1
    for layer in network:
        projection_count = 0
        for population in layer:
            projection_count += len(list(population))
        max_rows = max(max_rows, projection_count)

    fig1, axes = plt.subplots(max_rows, cols, figsize=(3.2*cols, 3.*max_rows))
    if max_rows == 1:
        if cols == 1:
            axes = [[axes]]
        else:
            axes = [axes]
    elif cols == 1:
        axes = [[axis] for axis in axes]
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            row = 0
            for population in layer:
                for projection in population:
                    this_axis = axes[row][col]
                    if projection.post is network.output_pop:
                        if sorted_output_idx is not None:
                            im = this_axis.imshow(projection.weight.data[sorted_output_idx, :], aspect='auto',
                                                  interpolation='none')
                        else:
                            im = this_axis.imshow(projection.weight.data, aspect='auto', interpolation='none')
                    elif projection.pre is network.output_pop:
                        if sorted_output_idx is not None:
                            im = this_axis.imshow(projection.weight.data[:, sorted_output_idx], aspect='auto',
                                                  interpolation='none')
                        else:
                            im = this_axis.imshow(projection.weight.data, aspect='auto', interpolation='none')
                    else:
                        im = this_axis.imshow(projection.weight.data, aspect='auto', interpolation='none')
                    fig1.colorbar(im, ax=this_axis)
                    this_axis.set_xlabel('Pre unit ID')
                    this_axis.set_ylabel('Post unit ID')
                    this_axis.set_title('%s.%s <- %s.%s' %
                              (projection.post.layer.name, projection.post.name,
                               projection.pre.layer.name, projection.pre.name))
                    row += 1
            while row < max_rows:
                this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig1.suptitle('%sweights' % title_str)
    fig1.tight_layout()
    fig1.show()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig2, axes = plt.subplots(max_rows, cols, figsize=(3.2 * cols, 3. * max_rows))
    if max_rows == 1:
        axes = [axes]
    for col, layer in enumerate(network):
        for row, population in enumerate(layer):
            this_axis = axes[row][col]
            if population is network.output_pop and sorted_output_idx is not None:
                im = this_axis.imshow(population.activity[:, sorted_output_idx].T, aspect='auto', interpolation='none')
            else:
                im = this_axis.imshow(population.activity.T, aspect='auto', interpolation='none')
            fig2.colorbar(im, ax=this_axis)
            this_axis.set_xlabel('Input pattern ID')
            this_axis.set_ylabel('Unit ID')
            this_axis.set_title('%s.%s' % (layer.name, population.name))
        row += 1
        while row < max_rows:
            this_axis = axes[row][col]
            this_axis.set_visible(False)
            row += 1
    fig2.suptitle('%sactivity' % title_str)
    fig2.tight_layout()
    fig2.show()

    cols = len(network.layers) - 1
    fig3, axes = plt.subplots(max_rows, cols, figsize=(3.2 * cols, 3. * max_rows))
    if max_rows == 1:
        if cols == 1:
            axes = [[axes]]
        else:
            axes = [axes]
    elif cols == 1:
        axes = [[axis] for axis in axes]
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                this_axis = axes[row][col]
                av_activity_dynamics = torch.mean(torch.stack(population.forward_steps_activity), dim=1)
                for i in range(population.size):
                    this_axis.plot(av_activity_dynamics[:,i])
                this_axis.set_xlabel('Equilibration time steps')
                this_axis.set_ylabel('Activity')
                this_axis.set_title('%s.%s' % (layer.name, population.name))
            row += 1
            while row < max_rows:
                this_axis = axes[row][col]
                this_axis.set_visible(False)
                row += 1
    fig3.suptitle('%sactivity dynamics' % title_str)
    fig3.tight_layout()
    fig3.show()


def plot_train_loss_history(network, title=None):
    """
    Plot loss and accuracy history from training
    :param network:
    :param title: str
    """
    # network.loss_history = network.loss_history.cpu()
    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)
    fig = plt.figure()
    plt.plot(network.loss_history)
    plt.ylabel('Train loss')
    plt.xlabel('Training steps')
    fig.suptitle('Train loss%s' % title_str)
    fig.tight_layout()
    fig.show()


def plot_validate_loss_history(network, title=None):
    """
    Assumes network has been trained and a val_loss_history has been stored.
    :param network:
    """
    assert len(network.val_loss_history) > 0, 'Network must contain a stored val_loss_history'

    if title is None:
        title_str = ''
    else:
        title_str = ': %s' % str(title)
    fig = plt.figure()
    plt.plot(network.val_history_train_steps, network.val_loss_history)
    plt.xlabel('Training steps')
    plt.ylabel('Validation loss')
    fig.suptitle('Validation loss%s' % title_str)
    fig.tight_layout()
    fig.show()


def evaluate_test_loss_history(network, test_dataloader, sorted_output_idx=None, store_history=False, plot=False):
    """
    Assumes network has been trained with store_params=True. Evaluates test_loss at each train step in the
    param_history.
    :param network:
    :param test_dataloader:
    :param sorted_output_idx: tensor of int
    :param store_history: bool
    :param plot: bool
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'
    assert len(network.param_history) > 0, 'Network must contain a stored param_history'

    idx, test_data, test_target = next(iter(test_dataloader))
    test_data = test_data.to(network.device)
    test_target = test_target.to(network.device)
    test_loss_history = []

    if store_history:
        network.reset_history()

    for state_dict in network.param_history:
        network.load_state_dict(state_dict)
        output = network.forward(test_data, store_history=store_history, no_grad=True)
        if sorted_output_idx is not None:
            output = output[:, sorted_output_idx]
        test_loss_history.append(network.criterion(output, test_target).item())

    network.test_loss_history = torch.stack(test_loss_history).cpu()

    fig = plt.figure()
    plt.plot(network.param_history_steps, network.test_loss_history)
    plt.xlabel('Training steps')
    plt.ylabel('Test loss')
    fig.suptitle('Test loss')
    fig.tight_layout()
    fig.show()


def plot_MNIST_examples(network, dataloader):
    """
    Test network performance on one example image from each MNIST class
    :param network:
    :param dataloader:
    """
    fig = plt.figure()
    axes = gs.GridSpec(nrows=1, ncols=10,
                       left=0.05, right=0.98,
                       wspace=0.5, hspace=0.5)

    # Compute accuracy on the test set
    indexes, images, targets = next(iter(dataloader))
    image_dim = int(math.sqrt(images.shape[1]))
    labels = torch.argmax(targets, axis=1)

    for label in range(10):
        ax = fig.add_subplot(axes[0, label])
        idx = torch.where(labels == label)[0][0]
        im = ax.imshow(images[idx].reshape((image_dim, image_dim)), cmap='Greys')
        ax.axis('off')
        output = network.forward(images[idx], no_grad=True)
        if labels[idx] == torch.argmax(output):
            color = 'k'
        else:
            color = 'red'
        ax.text(0, 35, f'pred.={torch.argmax(output)}', color=color)
    plt.suptitle('Example images',y=0.7)
    fig.show()


def plot_network_dynamics(network):
    """
    Plots activity dynamics for every population in the network
    """
    rows = len(network.layers)
    cols = np.max([len(layer.populations) for layer in network])

    fig = plt.figure(figsize=(8, 6))
    axes = gs.GridSpec(nrows=rows, ncols=cols,
                       left=0.05, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.8)

    for row, layer in enumerate(network):
        for col, population in enumerate(layer):
            ax = fig.add_subplot(axes[row, col])
            # average_activity_dynamics = torch.mean(population.activity_history, dim=0)
            average_activity_dynamics = torch.mean(population.activity_history[-21:], dim=(0,2))
            ax.plot(average_activity_dynamics)
            ax.set_title(f'{population.fullname} dynamics')


def plot_sparsity_history(network):
    rows = len(network.layers)
    cols = np.max([len(layer.populations) for layer in network])

    fig = plt.figure(figsize=(8, 6))
    axes = gs.GridSpec(nrows=rows, ncols=cols,
                       left=0.05, right=0.98,
                       top=0.83, bottom=0.1,
                       wspace=0.3, hspace=0.8)

    for row, layer in enumerate(network):
        for col, population in enumerate(layer):
            if 'Dend' not in population.name:
                ax = fig.add_subplot(axes[row, col])
                sparsity = torch.mean(population.sparsity_history[:,-1,:],dim=1)
                ax.plot(sparsity)
                ax.set_ylim(bottom=0.4,top=1)
                ax.set_title(f'{population.fullname} sparsity during training')


def plot_simple_EIANN_weight_history_diagnostic(network):
    fig, axes = plt.subplots(5, sharex=True)
    axes[0].plot(network.loss_history)
    axes[0].set_title('Loss')
    axes[1].plot(torch.mean(network.Output.E.H1.E.weight_history[:, :, 0], axis=1))
    axes[1].set_title('Output.E.H1.E')
    axes[2].plot(torch.mean(network.H1.E.Output.E.weight_history[:, :, 0], axis=1))
    axes[2].set_title('H1.E.Output.E')
    axes[3].plot(torch.mean(network.H1.E.Input.E.weight_history[:, :, 0], axis=1))
    axes[3].set_title('H1.E.Input.E')
    axes[4].plot(torch.mean(network.H1.E.H1.Dend_I.weight_history[:, :, 0], axis=1))
    axes[4].set_title('H1.E.H1.Dend_I')
    fig.show()


def plot_hidden_weights(weights, sort=False):
    num_rows = weights.shape[0]
    num_cols = int(num_rows ** 0.5)  # make the number of rows and columns approximately equal

    axes = gs.GridSpec(num_rows, num_cols)
    fig = plt.figure(figsize=(12, 12 * num_rows / num_cols))

    if sort: # Sort units by tuning structure of their receptive fields
        print("Computing tuning strength...")
        structure = utils.compute_rf_structure(weights.detach())
        sorted_idx = np.argsort(-structure)
        weights = weights[sorted_idx]

    print("Generating plots...")
    for i, unit_weight_vec in enumerate(weights):
        # Calculate the row and column indices for the current subplot
        row_idx = i // num_cols
        col_idx = i % num_cols

        img_dim = int(unit_weight_vec.shape[0] ** 0.5)
        img = unit_weight_vec.view(img_dim, img_dim).detach().cpu()

        # Add a subplot to the figure at the specified row and column
        ax = fig.add_subplot(axes[row_idx, col_idx])
        im = ax.imshow(img, cmap='gray', vmin=torch.min(weights), vmax=torch.max(weights))
        ax.axis('off')

    print(f"W_min = {torch.min(weights)}, W_max = {torch.max(weights)}")
    fig.tight_layout(pad=0.2)
    fig_height = fig.get_size_inches()[1]
    cax = fig.add_axes([0.005, ax.get_position().y0-0.2/fig_height, 0.5, 0.12/fig_height])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')


def plot_receptive_fields(receptive_fields, activity_preferred_inputs=None, sort=False, num_cols=None, num_rows=None,
                          activity_threshold=1e-10, cmap='bwr'):
    """
    Plot receptive fields of hidden units, optionally weighted by their activity. Units are sorted by their tuning
    structure. The receptive fields are normalized so the max=1 (while values at 0 are preserved). The colormap is
    custom so that negative values are blue and positive values are gray.

    :param receptive_fields:
    :param activity_preferred_inputs:
    :param sort:
    :param activity_threshold: float
    :param num_cols: int
    :param num_rows: int
    """
    if activity_preferred_inputs is not None:
        activity_preferred_inputs = activity_preferred_inputs.diagonal()
        print(f'Min activity: {torch.min(activity_preferred_inputs)}, Max activity: {torch.max(activity_preferred_inputs)}')
        active_idx = torch.where(activity_preferred_inputs > activity_threshold)
        activity_preferred_inputs = activity_preferred_inputs[active_idx]
        receptive_fields = receptive_fields[active_idx]

    if sort: # Sort units by tuning structure of their receptive fields
        structure = utils.compute_rf_structure(receptive_fields)
        sorted_idx = np.argsort(-structure)
        receptive_fields = receptive_fields[sorted_idx]
        if activity_preferred_inputs is not None:
            activity_preferred_inputs = activity_preferred_inputs[sorted_idx]

    if activity_preferred_inputs is not None:
        # Normalize each receptive_field so the max=1 (while values at 0 are preserved)
        receptive_fields = receptive_fields / (torch.max(receptive_fields, dim=1, keepdim=True)[0] + 1e-10)
        receptive_fields = receptive_fields * activity_preferred_inputs.unsqueeze(1)

    # Create figure
    num_units = receptive_fields.shape[0]
    if num_cols is None:
        if num_units < 25:
            num_cols = 5
        else:
            num_cols = int(np.ceil(num_units**0.5))
    if num_rows is None:
        num_rows = int(np.ceil(num_units / num_cols))
    receptive_fields = receptive_fields[:num_cols * num_rows]

    size = num_cols
    # size = 6
    num_rows += 1
    axes = gs.GridSpec(num_rows, num_cols)
    fig = plt.figure(figsize=(size, size * num_rows / num_cols))

    colorscale_max = torch.max(receptive_fields.abs())
    if torch.min(receptive_fields)<0 and cmap is None:
        # Create custom colormap
        top_rgba = plt.get_cmap('gray')(np.linspace(0, 1, 256))
        bottom_rgba = plt.get_cmap('Blues')(np.linspace(0, 1, 256))
        colors = np.concatenate((bottom_rgba, top_rgba))

        my_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors)
        colorscale_min = -colorscale_max
    elif cmap is None: # all positive values
        my_cmap = 'gray'
        colorscale_min = 0
    elif cmap is not None:
        my_cmap = cmap
        colorscale_min = -colorscale_max

    for i in range(receptive_fields.shape[0]):
        row_idx = i // num_cols
        col_idx = i % num_cols

        ax = fig.add_subplot(axes[row_idx, col_idx])
        im = ax.imshow(receptive_fields[i].view(28, 28), cmap=my_cmap, vmin=colorscale_min, vmax=colorscale_max,
                       aspect='equal')
        ax.axis('off')

    fig.tight_layout(pad=2., h_pad=0.4, w_pad=0.2)
    fig_width, fig_height = fig.get_size_inches()
    cax = fig.add_axes([0.03, ax.get_position().y0-0.2/fig_height, 0.5, 0.12/fig_height])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    fig.show()

    # fig_height = fig.get_size_inches()[1]
    # wspace = 0.01; hspace = 0.01; left = 0.01; top = 0.99; right = 0.99; bottom = 0.1
    # # cax = fig.add_axes([left, ax.get_position().y0-0.2/fig_height, 0.5, 0.12/fig_height])
    # cax = fig.add_subplot(axes[row_idx + 1, :num_cols//2])
    # cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    # plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, top=top, right=right, bottom=bottom)


def plot_unit_receptive_field(population, dataloader, unit):

    fig, ax = plt.subplots(1,2, figsize=(6,3))

    weighted_avg_input = utils.compute_act_weighted_avg(population, dataloader)
    unit_receptive_field = weighted_avg_input[unit]
    ax[0].imshow(unit_receptive_field.view(28, 28), cmap='gray')
    ax[0].set_title('Weighted Average Input')
    ax[0].axis('off')

    unit_receptive_field = utils.compute_unit_receptive_field(population, dataloader, unit)
    ax[1].imshow(unit_receptive_field.view(28, 28), cmap='gray')
    ax[1].set_title('Activation Maximization')
    ax[1].axis('off')


def plot_hidden_weight_history(network, unit=0):
    '''
    Plot a single unit's receptive field over time
    '''

    weight_history = network.module_dict['H1E_InputE'].weight_history[:, unit, :]
    unit_plateaus = network.H1.E.plateau_history[:, unit]
    plateau_scale = max(abs(max(unit_plateaus)), abs(min(unit_plateaus)))
    label_history = torch.argmax(torch.stack(network.target_history), dim=1)

    num_rows = weight_history.shape[0]
    num_cols = 15
    fig = plt.figure(figsize=(12, 12 * num_rows / num_cols))
    axes = gs.GridSpec(num_rows, num_cols,
                       top=0.975, wspace=0.1)

    max_step = min(150, weight_history.shape[0])
    step_range = np.arange(0, max_step, 1)
    for i, train_step in enumerate(tqdm(step_range)):
        row_idx = i // num_cols
        col_idx = i % num_cols

        weight_vec = weight_history[train_step]

        ax = fig.add_subplot(axes[row_idx, col_idx])
        im = ax.imshow(weight_vec.view(28, 28), cmap='gray',
                       vmin=torch.min(weight_history), vmax=torch.max(weight_history))
        ax.axis('off')

        if train_step <= unit_plateaus.shape[0] and unit_plateaus[train_step] != 0:
            ax.scatter([24], [3], s=30, c=unit_plateaus[train_step],
                       cmap='bwr', vmin=-plateau_scale, vmax=plateau_scale)
            ax.text(0, 5, s=str(int(label_history[train_step])), fontsize=10, color='w')

        if i < num_cols and i % 2 == 0:
            ax.set_title(f'step {network.param_history_steps[i]}', fontsize=9)

    steps1 = network.param_history_steps[step_range[0]]
    steps2 = network.param_history_steps[step_range[-1]]
    print(f"Input->H1 weight history: Unit {unit}, steps {steps1}-{steps2}")
    print(f"W_min = {torch.min(weight_history)}, W_max = {torch.max(weight_history)}")


def plot_binary_decision_boundary(network, test_dataloader, hard_boundary=False, num_points = 1000):
    assert len(test_dataloader)==1, "Dataloader must have a single large batch"

    # Compute accuracy on test data
    idx, data, target = next(iter(test_dataloader))
    output = network(data)
    prediction = torch.heaviside(output - 0.5, torch.tensor(0.))
    test_accuracy = 100 * torch.sum(prediction == target) / len(target)
    print(f"Test Accuracy {test_accuracy} %")

    # Plot decision boundary (test a grid of X values)
    extension = 2.
    x1 = data[:, 0].numpy()
    x2 = data[:, 1].numpy()
    # x1_extension = (np.max(x1) - np.min(x1)) * extension
    # x2_extension = (np.max(x2) - np.min(x2)) * extension
    x1_extension = 1.5
    x2_extension = 1.5
    x1_range = np.linspace(np.min(x1) - x1_extension, np.max(x1) + x1_extension, num_points)
    x2_range = np.linspace(np.min(x2) - x2_extension, np.max(x2) + x2_extension, num_points)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

    flat_x1_vals = x1_mesh.reshape(1, num_points ** 2)
    flat_x2_vals = x2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_x1_vals, flat_x2_vals]).T

    meshgrid_points = torch.tensor(meshgrid_points).type(torch.float32)
    outputs = network(meshgrid_points).detach()
    output_grid = outputs.reshape([x1_range.size, x2_range.size]).flipud()

    if hard_boundary:
        output_grid = np.heaviside(output_grid-0.5,0)

    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange", "white", "blue"])
    ax.scatter(data[:, 0], data[:, 1], c=target, edgecolors='w', cmap=cmap)
    im = plt.imshow(output_grid, cmap=cmap, vmin=0, vmax=1, alpha=0.6, interpolation='nearest',
                    extent=[np.min(x1_range), np.max(x1_range), np.min(x2_range), np.max(x2_range)])
    cax = fig.add_axes([ax.get_position().x1 + 0.04, ax.get_position().y0, 0.03, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    fig.show()


def plot_batch_accuracy(network, test_dataloader, population=None, sorted_output_idx=None, title=None):
    """
    Compute total accuracy (% correct) on given dataset
    :param network:
    :param test_dataloader:
    :param population: :class:'Population' or str 'all'
    :param sorted_output_idx: tensor of int
    :param title: str
    """
    assert len(test_dataloader)==1, 'Dataloader must have a single large batch'

    idx, data, targets = next(iter(test_dataloader))
    data = data.to(network.device)
    targets = targets.to(network.device)
    labels = torch.argmax(targets, axis=1)  # convert from 1-hot vector to int label
    output = network.forward(data, no_grad=True)

    # if unsupervised: # sort output units by their mean activity
    if sorted_output_idx is not None:
        output = output[:, sorted_output_idx]
    percent_correct = 100 * torch.sum(torch.argmax(output, dim=1) == labels) / data.shape[0]
    percent_correct = torch.round(percent_correct, decimals=2)
    print(f'Batch accuracy = {percent_correct}%')

    # Plot average output for each label class
    num_units = targets.shape[1]
    num_labels = num_units
    avg_output = torch.zeros(num_units, num_labels)

    for label in range(num_labels):
        label_idx = torch.where(labels == label)  # find all instances of given label
        avg_output[:, label] = torch.mean(output[label_idx], dim=0)

    fig, axes = plt.subplots()
    ax = axes
    im = ax.imshow(avg_output, aspect='auto', interpolation='none')
    cbar = plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_labels))
    ax.set_yticks(range(num_units))
    ax.set_xlabel('Labels')
    ax.set_ylabel('Output unit')
    if title is not None:
        ax.set_title(f'Average activity - {network.output_pop.fullname}\n{title}')
    else:
        ax.set_title(f'Average activity - {network.output_pop.fullname}')
    fig.tight_layout()
    fig.show()

    pop_list = []
    if population == 'all':
        for layer in network:
            for pop in layer:
                if pop is not network.output_pop:
                    pop_list.append(pop)
    elif population is not None and population is not network.output_pop:
        pop_list.append(population)

    for population in pop_list:
        avg_pop_activity = torch.zeros(population.size, num_labels)
        for label in range(num_labels):
            label_idx = torch.where(labels == label)  # find all instances of given label
            avg_pop_activity[:, label] = torch.mean(population.activity[label_idx], dim=0)
        silent_unit_indexes = torch.where(torch.sum(avg_pop_activity, dim=1) == 0)[0]
        active_unit_indexes = torch.where(torch.sum(avg_pop_activity, dim=1) > 0)[0]
        preferred_input = torch.argmax(avg_pop_activity[active_unit_indexes], dim=1)
        _, idx = torch.sort(preferred_input)
        sort_idx = torch.concat([active_unit_indexes[idx], silent_unit_indexes])
        fig, axes = plt.subplots()
        ax = axes
        im = ax.imshow(avg_pop_activity[sort_idx], interpolation='none', aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        ax.set_xticks(range(num_labels))
        ax.set_xlabel('Labels')
        ax.set_ylabel(f'{population.fullname} unit')
        if title is not None:
            ax.set_title(f'Average activity - {population.fullname}\n{title}')
        else:
            ax.set_title(f'Average activity - {population.fullname}')
        fig.tight_layout()
        fig.show()


def plot_rsm(network, test_dataloader):

    idx, data, target = next(iter(test_dataloader))
    data.to(network.device)
    network.forward(data, no_grad=True)
    pop_activity = network.H1.E.activity

    # Sort rows of pop_activity by label
    _, sort_idx = torch.sort(torch.argmax(target, dim=1))
    pop_activity = pop_activity[sort_idx]

    similarity_matrix = cosine_similarity(pop_activity)

    fig, ax = plt.subplots()
    im = plt.imshow(similarity_matrix, interpolation='none')
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Cosine similarity', rotation=270, labelpad=15)

    num_samples = target.shape[0]
    num_labels = target.shape[1]
    samples_per_label = num_samples // num_labels
    x_ticks = np.arange(samples_per_label / 2, num_samples, samples_per_label)
    y_ticks = np.arange(samples_per_label / 2, num_samples, samples_per_label)
    ax.set_xticklabels(range(0, num_labels))
    ax.set_yticklabels(range(0, num_labels))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlabel('Patterns (sorted by label)')
    ax.set_ylabel('Patterns (sorted by label)')
    ax.set_title('Representational Similarity Matrix')


def plot_plateaus(population, start=0, end=-1):
    fig = plt.figure(figsize=[11,7])
    plateaus = population.plateau_history[start:end].T
    plt.imshow(plateaus, aspect='auto',interpolation='None',
              vmin=-1,vmax=1,cmap='bwr')

    plt.ylabel(f'{population.fullname} unit')
    plt.xlabel('Train step')
    plt.title(f'Plateau History: step {start} to {end}')


def plot_sorted_plateaus(population, test_dataloader, show_negative=True):

    network = population.network

    # Sort history (x-axis) by label
    # TODO: Why was this torch.stack previously?
    label_history = torch.argmax(network.target_history, dim=1)
    val, idx = torch.sort(label_history)
    sorted_plateaus = population.plateau_history[idx, :]

    # Sort units (y-axis) by preferred input, defined over test_dataloader
    idx, data, target = next(iter(test_dataloader))
    network.forward(data, no_grad=True) #compute activities for test dataset
    labels = torch.argmax(target, dim=1)
    num_labels = target.shape[1]
    samples_per_label = torch.zeros(num_labels)
    avg_pop_activity = torch.zeros(population.size, num_labels)
    for label in range(num_labels):
        label_idx = torch.where(labels == label)[0]
        samples_per_label[label] = len(label_idx)
        avg_pop_activity[:, label] = torch.mean(population.activity[label_idx,:], dim=0)
    silent_unit_indexes = torch.where(torch.sum(avg_pop_activity, dim=1) == 0)[0]
    active_unit_indexes = torch.where(torch.sum(avg_pop_activity, dim=1) > 0)[0]
    preferred_input = torch.argmax(avg_pop_activity[active_unit_indexes], dim=1)
    val, idx = torch.sort(preferred_input)
    sorted_indexes = torch.concat([active_unit_indexes[idx], silent_unit_indexes])
    sorted_plateaus = sorted_plateaus[:, sorted_indexes]
    if not show_negative:
        sorted_plateaus[(sorted_plateaus < 0)] = 0
        vmin = 0
        cmap = 'binary'
    else:
        vmin = -1
        cmap = 'bwr'
    unit_ids = idx

    fig, ax = plt.subplots() # figsize=[11, 7])
    # plateau_scale = max(abs(torch.max(sorted_plateaus)), abs(torch.min(sorted_plateaus)))
    ax.imshow(sorted_plateaus.T, aspect='auto', cmap=cmap, interpolation='nearest',
               vmin=vmin, vmax=1)
    ax.set_ylabel(f'{population.fullname} unit')
    ax.set_title(f'Sorted Plateau History, {population.fullname}')

    label_centers = torch.cumsum(samples_per_label, dim=0) - samples_per_label // 2
    ax.set_xticks(label_centers)
    ax.set_xticklabels(range(0, num_labels))
    ax.set_xlabel('Samples (sorted by label)')
    fig.tight_layout()
    clean_axes(ax)
    fig.show()

    return sorted_plateaus.T, unit_ids


def plot_total_input(population, test_dataloader, sorting='E', act_threshold=0):
    """
    Plot the total input to a population for each pattern in the test set
    TODO: check activity_history dimensions
    :param population:
    :param test_dataloader:
    :param sorting:
    :param act_threshold:
    :return:
    """
    ''''''

    network = population.network
    network.reset_history()
    idx, data, target = next(iter(test_dataloader))
    network.forward(data, store_history=True, no_grad=True)

    total_input = {}
    for name, projection in population.incoming_projections.items():
        if projection.direction in ['F','forward']:
            total_input[name] = projection.weight.data @ projection.pre.activity_history[0,-1,:,:].data.T
        elif projection.direction in ['R','recurrent']:
            pre_act = torch.mean(projection.pre.activity_history[0,-4:,:,:].data, dim=0).data.T
            total_input[name] = projection.weight.data @ pre_act

    if sorting == 'E':
        val, idx = torch.sort(total_input['H1E_InputE'], descending=True)
    elif sorting == 'I':
        val, idx = torch.sort(total_input['H1E_H1FBI'], descending=False)
    elif sorting == 'EI_balance':
        net_input = total_input['H1E_InputE'] + total_input['H1E_H1FBI']
        val, idx = torch.sort(net_input, descending=True)
    elif sorting == 'activity':
        activity = population.activity.T
        val, idx = torch.sort(activity, descending=True)

    fig, ax = plt.subplots(1, 2, figsize=[11, 4])

    for name, proj_input in total_input.items():
        if torch.mean(proj_input) > 0:
            color = 'r'
        else:
            color = 'C0'
        sorted_input = torch.gather(proj_input, 1, idx)
        active_units = torch.where(torch.sum(population.activity, dim=0) > act_threshold)[0]
        # net_input = total_input['H1E_InputE'] + total_input['H1E_H1FBI']
        # active_units = torch.where(torch.max(net_input, dim=1)[0] > act_threshold)[0]
        avg_proj_input = torch.mean(sorted_input[active_units], dim=0)
        ax[0].plot(np.abs(avg_proj_input), label=name, c=color, alpha=0.8)
    ax[0].set_xlabel('Input pattern')
    ax[0].set_ylabel('Weighted input (abs)')
    ax[0].set_title(f'Average total E/I input to {population.fullname} (sorted)')
    ax[0].legend()

    # net_input = total_input['H1E_InputE'] + total_input['H1E_H1FBI']
    # active_idx = torch.where(net_input > act_threshold)
    active_idx = torch.where(population.activity.T > act_threshold)
    ax[1].scatter(total_input['H1E_InputE'][active_idx], total_input['H1E_H1FBI'][active_idx], c='k', alpha=0.2)
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Sample E input')
    ax[1].set_ylabel('Sample I input')
    ax[1].set_title(f'Total E/I input to {population.fullname}')
    fig.tight_layout()


def plot_correlations(network, test_dataloader):
    '''Plot the correlation matrix for a layer's weights'''

    idx, data, target = next(iter(test_dataloader))
    network.forward(data, no_grad=True)

    for layer in network:
        if layer.name == 'Input':
            continue

        # Compute correlations
        activity_matrix = torch.cat([layer.E.activity.T, layer.SomaI.activity.T])
        activity_corr_mat = cosine_similarity(activity_matrix)

        E_I_weights = network.module_dict[f"{layer.E.fullname}_{layer.SomaI.fullname}"].weight.data
        I_E_weights = network.module_dict[f"{layer.SomaI.fullname}_{layer.E.fullname}"].weight.data

        # Generate plots
        fig, ax = plt.subplots(1, 4, figsize=[14, 3])

        plot_nr = 0
        ax[plot_nr].scatter(I_E_weights, E_I_weights.T, c='k', alpha=0.2)
        ax[plot_nr].invert_yaxis()
        ax[plot_nr].set_title('Weight correlations')
        ax[plot_nr].set_xlabel('E->SomaI weights')
        ax[plot_nr].set_ylabel('SomaI->E weights')
        m, b = np.polyfit(I_E_weights.flatten(), E_I_weights.T.flatten(), 1)
        x = np.linspace(0, torch.max(I_E_weights))
        y = m * x + b
        ax[plot_nr].plot(x, y, '--', c='red', linewidth=3)
        # print(f'Linear regression: y = {m} x + {b}')
        r_val, p_val = stats.pearsonr(I_E_weights.flatten(), E_I_weights.T.flatten())
        plot_nr += 1

        im = ax[plot_nr].imshow(activity_corr_mat[0:layer.E.size, 0:layer.E.size],
                                vmin=np.min(activity_corr_mat), vmax=np.max(activity_corr_mat))
        plt.colorbar(im, ax=ax[plot_nr])
        ax[plot_nr].set_xticks([])
        ax[plot_nr].set_yticks([])
        ax[plot_nr].set_ylabel('E units')
        ax[plot_nr].set_xlabel('E units')
        ax[plot_nr].set_title(f'{layer.E.fullname} ')
        plot_nr += 1

        im = ax[plot_nr].imshow(activity_corr_mat[layer.E.size:, 0:layer.E.size],
                                vmin=np.min(activity_corr_mat), vmax=np.max(activity_corr_mat))
        plt.colorbar(im, ax=ax[plot_nr])
        ax[plot_nr].set_xticks([])
        ax[plot_nr].set_yticks([])
        ax[plot_nr].set_ylabel('I units')
        ax[plot_nr].set_xlabel('E units')
        ax[plot_nr].set_title(f'E/I correlation')
        plot_nr += 1

        im = ax[plot_nr].imshow(activity_corr_mat[layer.E.size:, layer.E.size:],
                                vmin=np.min(activity_corr_mat), vmax=np.max(activity_corr_mat))
        ax[plot_nr].set_title(f'{layer.SomaI.fullname} ')
        plt.colorbar(im, ax=ax[plot_nr])
        ax[plot_nr].set_xticks([])
        ax[plot_nr].set_yticks([])
        ax[plot_nr].set_ylabel('I units')
        ax[plot_nr].set_xlabel('I units')
        plot_nr += 1

        plt.suptitle(f'{layer.name} activity similarity matrix', fontsize=18)

        plt.tight_layout(pad=0.5)
        plt.show()
        print(f'Pearson correlation: r={r_val:.3f}, r^2={r_val ** 2:.3f}, p={p_val:.2E}')

        # Plot weight vs. activity correlation
        EI_corr = activity_corr_mat[layer.E.size:, 0:layer.E.size]
        fig, ax = plt.subplots(1, 2, figsize=[8, 2])
        ax[0].scatter(EI_corr, I_E_weights, c='r', alpha=0.2)
        ax[0].set_xlabel('E/I activity correlation')
        ax[0].set_ylabel('Weight')
        m, b = np.polyfit(EI_corr.flatten(), I_E_weights.flatten(), 1)
        x = np.linspace(np.min(EI_corr), np.max(EI_corr), 10)
        y = m * x + b
        ax[0].plot(x, y, '--', c='k', linewidth=2)
        # print(f'Linear regression: y = {m} x + {b}')
        r_val, p_val = stats.pearsonr(EI_corr.flatten(), I_E_weights.flatten())
        txt_E = f'Pearson correlation (E): r={r_val:.3f}, r^2={r_val ** 2:.3f}, p={p_val:.2E}'

        ax[1].scatter(EI_corr, E_I_weights.T, c='b', alpha=0.2)
        ax[1].set_xlabel('E/I activity correlation')
        ax[1].set_ylabel('Weight')
        ax[1].invert_yaxis()
        m, b = np.polyfit(EI_corr.flatten(), E_I_weights.T.flatten(), 1)
        y = m * x + b
        ax[1].plot(x, y, '--', c='k', linewidth=2)
        # print(f'Linear regression: y = {m} x + {b}')
        r_val, p_val = stats.pearsonr(EI_corr.flatten(), E_I_weights.T.flatten())
        txt_I = f'Pearson correlation (I): r={r_val:.3f}, r^2={r_val ** 2:.3f}, p={p_val:.2E}'
        plt.tight_layout()
        plt.show()
        print(txt_E)
        print(txt_I)



# *******************************************************************
# Loss landscape functions
# *******************************************************************
def plot_weight_history_PCs(network):
    """
    Function performs PCA on a given set of weights and
        1. plots the explained variance
        2. the trajectory of the weights in the PC space during the course of learning
        3. the loading scores of the weights

    Parameters
    weight_history: torch tensor, size: [time_steps x total number of weights]

    Returns
    """
    flat_weight_history,_ = get_flat_weight_history(network)

    # Center the data (mean=0, std=1)
    w_mean = torch.mean(flat_weight_history, axis=0)
    w_std = torch.std(flat_weight_history, axis=0)
    flat_weight_history = (flat_weight_history - w_mean) / (w_std + 1e-10) # add epsilon to avoid NaNs

    pca = PCA(n_components=5)
    pca.fit(flat_weight_history)
    weights_pca_space = pca.transform(flat_weight_history)

    # Plot explained variance
    fig, ax = plt.subplots(1, 3)
    explained_variance = pca.explained_variance_ratio_
    percent_exp_var = np.round(explained_variance * 100, decimals=2)
    labels = ['PC' + str(x) for x in range(1, len(percent_exp_var) + 1)]
    ax[0].bar(x=range(1, len(percent_exp_var) + 1), height=percent_exp_var, tick_label=labels)
    ax[0].set_ylabel('Percentage of variance explained')
    ax[0].set_xlabel('Principal Component')
    ax[0].set_title('Scree Plot')

    # Plot weights in PC space
    PC1 = weights_pca_space[:, 0]
    PC2 = weights_pca_space[:, 1]
    ax[1].scatter(PC1, PC2)
    ax[1].scatter(PC1[0], PC2[0], color='blue', label='before training')
    ax[1].scatter(PC1[-1], PC2[-1], color='red', label='after training')
    ax[1].set_xlabel(f'PC1 - {percent_exp_var[0]}%')
    ax[1].set_ylabel(f'PC2 - {percent_exp_var[1]}%')
    ax[1].legend()

    # Plot loading scores for PC1 to determine which/how many weights are important for variance along PC1
    sorted_loadings = -np.sort(-np.abs(pca.components_[0]))  # Loadings sorted in descending order of abs magnitude
    sorted_idx = np.argsort(-np.abs(pca.components_[0]))

    most_important_weights_flat = sorted_idx[0:10]  #
    most_important_weights_idx = []  # index of important weights in original weight matrix

    ax[2].plot(sorted_loadings)
    ax[2].set_xlabel('sorted weights')
    ax[2].set_ylabel('Loading \n(alignment with weight)')
    ax[2].set_title('PC1 weight components')

    plt.tight_layout()
    # fig.show()


def plot_param_history_PCs(flat_param_history):
    """
    Function performs PCA on a given set of parameters (drawn from the network state_dict()) and
        1. plots the explained variance
        2. the trajectory of the weights in the PC space during the course of learning
        3. the loading scores of the weights

    Parameters
    flat_param_history: torch tensor, size: [time_steps x total number of parameters]

    Returns
    """
    # Center the data (mean=0, std=1)
    p_mean = torch.mean(flat_param_history, axis=0)
    p_std = torch.std(flat_param_history, axis=0)
    flat_param_history = (flat_param_history - p_mean) / (p_std + 1e-10)  # add epsilon to avoid NaNs

    pca = PCA(n_components=5)
    pca.fit(flat_param_history)
    params_pca_space = pca.transform(flat_param_history)

    # Plot explained variance
    fig, ax = plt.subplots(1, 3)
    explained_variance = pca.explained_variance_ratio_
    percent_exp_var = np.round(explained_variance * 100, decimals=2)
    labels = ['PC' + str(x) for x in range(1, len(percent_exp_var) + 1)]
    ax[0].bar(x=range(1, len(percent_exp_var) + 1), height=percent_exp_var, tick_label=labels)
    ax[0].set_ylabel('Percentage of variance explained')
    ax[0].set_xlabel('Principal Component')
    ax[0].set_title('Scree Plot')

    # Plot weights in PC space
    PC1 = params_pca_space[:, 0]
    PC2 = params_pca_space[:, 1]
    ax[1].scatter(PC1, PC2)
    ax[1].scatter(PC1[0], PC2[0], color='blue', label='before training')
    ax[1].scatter(PC1[-1], PC2[-1], color='red', label='after training')
    ax[1].set_xlabel(f'PC1 - {percent_exp_var[0]}%')
    ax[1].set_ylabel(f'PC2 - {percent_exp_var[1]}%')
    ax[1].legend()

    # Plot loading scores for PC1 to determine which/how many weights are important for variance along PC1
    sorted_loadings1 = -np.sort(-np.abs(pca.components_[0]))  # Loadings sorted in descending order of abs magnitude
    sorted_idx = np.argsort(-np.abs(pca.components_[0]))

    most_important_weights_flat = sorted_idx[0:10]  #
    most_important_weights_idx = []  # index of important weights in original weight matrix

    ax[2].plot(sorted_loadings1)
    ax[2].set_xlabel('sorted weights')
    ax[2].set_ylabel('Loading \n(alignment with weight)')
    ax[2].set_title('PC1 weight components')

    plt.tight_layout()

    return fig, sorted_loadings1


def get_flat_param_history(param_history):
    """
    Flattens all parameters (weights, biases, and other registered paramters)
    into a single vector for every point in the training history

    :param: network: EIANN network
    :return: matrix of flattened parameter vectors for every point in the training hisotry
    :return: list containing tuples of (num weights, weight matrix shape)
    """

    flat_param_history = []
    param_metadata = {}

    for i, state_dict in enumerate(param_history):
        param_vector = []
        for name, param in state_dict.items():
            param_vector.append(param.flatten())
            if i == 0:
                param_metadata[name] = (param.numel(), param.shape)
        flat_param_history.append(torch.cat(param_vector))
    flat_param_history = torch.stack(flat_param_history)

    return flat_param_history, param_metadata


def get_flat_weight_history(network):
    """
    Flattens all weights into a single vector for every point in the training history

    :param: network: EIANN network with weights stored as Projection attributes
    :return: tuple: matrix of flattened weight vectors for every point in the training hisotry
                    list containing tuples of (num weights, weight matrix shape)
    """
    flat_weight_history = []
    weight_sizes = []
    for layer in network:
        for population in layer:
            for projection in population:
                W_hist = projection.weight_history
                flat_weight_history.append(W_hist.flatten(1))
                weight_sizes.append([W_hist[0].numel(), W_hist[0].shape])

    flat_weight_history = torch.cat(flat_weight_history,dim=1)

    return flat_weight_history, weight_sizes


def flatten_weights(network):
    """
    Flattens all weights into a single vector.

    :param: network: EIANN_archive network with weights stored as Projection attributes
    :return: flattened weight vector (containing all network weights)
    :return: list containing tuples of (num weights, weight matrix shape)
    """

    flat_weights_ls = []
    weight_sizes = []

    for layer in network:
        for population in layer:
            for projection in population:
                W = projection.weight.data
                flat_weights_ls.append(W.flatten())
                weight_sizes.append([W.numel(), W.shape])

    flat_weights = torch.cat(flat_weights_ls)

    return flat_weights, weight_sizes


def unflatten_weights(flat_weights, weight_sizes):
    """
    Convert flat weight vector to list of tensors (with correct dimensions)

    :param: flat_weights: flat vector of weights
    :param: weight_sizes: list containing tuples of (num weights, weight matrix shape)
    :return: list containing reshaped weight matrices
    """
    weight_mat_ls = []
    idx_start = 0
    for length, shape in weight_sizes:
        w = flat_weights[idx_start:idx_start + length]
        weight_mat_ls.append(w.reshape(shape).type(torch.float32))
        idx_start += length

    return weight_mat_ls


def unflatten_params(flat_params, param_metadata):
    """
    Convert flat vector of parameters to list of tensors (with correct dimensions)

    :param: flat_params: flat vector of parameters
    :param: param_metadata: list containing tuples of (num params, param dimensions)
    :return: list containing reshaped param matrices
    """
    state_dict = {}
    idx_start = 0
    for param_name in param_metadata:
        length, shape = param_metadata[param_name]
        p = flat_params[idx_start:idx_start + length]
        state_dict[param_name] = p.reshape(shape).type(torch.float32)
        idx_start += length

    return state_dict


def compute_loss(network, state_dict, test_dataloader):
    """
    Calculate the loss for the network given a specified set of parameters and test dataset

    :param: network: EIANN_archive network
    :param: state_dict: network state dict containing desired parameters to test
    :param: test_dataloader: dataloader with (data,target) to use for computing loss.
    :return: loss
    """
    # Insert weight matrices into network
    network.load_state_dict(state_dict)

    # Compute loss on dataset
    loss = 0
    for batch in test_dataloader:
        idx, batch_data, batch_target = batch
        output = network.forward(batch_data, no_grad=True)
        loss += network.criterion(output, batch_target).item()
    loss /= len(test_dataloader)

    return loss


def plot_loss_landscape(test_dataloader, network1, network2=None, num_points=20, extension=0.2, vmax=1.2, plot_line_loss=False):
    """
    Plots the loss landscape (with respect to the given test dataloader) of a network in the PC space defined by the first two 
    principal components of the weight history. The loss is computed for a grid of points in the PC space, as well as for the 
    weight trajectory of the network through the landscape.

    :param test_dataloader: dataloader with (data,target) to use for computing loss.
    :param network1: EIANN network
    :param network2: EIANN network
    :param num_points: number of points in each dimension of the PC space grid
    :param extension: fraction of PC space to extend the grid beyond the range of the weight history
    :param vmax: maximum value for the loss heatmap
    :param plot_line_loss: if True, plot the loss for the weight history of the network
    
    :return:
    """
    flat_param_history, param_metadata = get_flat_param_history(network1.param_history)
    history_len1 = flat_param_history.shape[0]

    if network2 is not None:
        flat_param_history2, param_metadata2 = get_flat_param_history(network2.param_history)
        flat_param_history = torch.cat([flat_param_history, flat_param_history2])
        assert param_metadata == param_metadata2, "Network architecture must match"

    # Center the data (mean=0, std=1)
    p_mean = torch.mean(flat_param_history, axis=0)
    p_std = torch.std(flat_param_history, axis=0)
    flat_param_history = (flat_param_history - p_mean) / (p_std + 1e-10)  # add epsilon to avoid NaNs

    # Get weights in gridplane defined by PC dimensions
    pca = PCA(n_components=2)
    pca.fit(flat_param_history)
    param_hist_pca_space = pca.transform(flat_param_history)

    explained_variance = pca.explained_variance_ratio_
    percent_exp_var = np.round(explained_variance * 100, decimals=2)

    PC1 = param_hist_pca_space[:, 0]
    PC2 = param_hist_pca_space[:, 1]
    PC1_extension = (np.max(PC1) - np.min(PC1)) * extension
    PC2_extension = (np.max(PC2) - np.min(PC2)) * extension
    PC1_range = np.linspace(np.min(PC1) - PC1_extension, np.max(PC1) + PC1_extension, num_points)
    PC2_range = np.linspace(np.min(PC2) - PC2_extension, np.max(PC2) + PC2_extension, num_points)
    PC1_mesh, PC2_mesh = np.meshgrid(PC1_range, PC2_range)

    # Convert PC coordinates into full weight vectors
    flat_PC1_vals = PC1_mesh.reshape(1, num_points ** 2)
    flat_PC2_vals = PC2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_PC1_vals, flat_PC2_vals]).T

    gridpoints_paramspace = pca.inverse_transform(meshgrid_points)
    gridpoints_paramspace = torch.tensor(gridpoints_paramspace) * p_std + p_mean

    # Compute loss for points in grid
    test_network = copy(network1)  # create copy to avoid modifying original networks
    losses = torch.zeros(num_points ** 2)
    for i, gridpoint_flat in enumerate(tqdm(gridpoints_paramspace)):
        state_dict = unflatten_params(gridpoint_flat, param_metadata)
        losses[i] = compute_loss(test_network, state_dict, test_dataloader)
    loss_grid = losses.reshape([PC1_range.size, PC2_range.size])

    # plot_3D_loss_surface(loss_grid, PC1_mesh, PC2_mesh)

    if network2 is not None:
        PC1_network1 = PC1[0:history_len1]
        PC2_network1 = PC2[0:history_len1]

        PC1_network2 = PC1[history_len1:]
        PC2_network2 = PC2[history_len1:]

    if plot_line_loss:
        fig = plt.figure()

        # Compute loss for points in weight history
        loss_history = torch.zeros(flat_param_history.shape[0])
        for i, flat_params in enumerate(tqdm(flat_param_history)):
            state_dict = unflatten_params(flat_params, param_metadata)
            loss_history[i] = compute_loss(network1, state_dict, test_dataloader)

        # Plot loss heatmap
        vmax = torch.max(torch.cat([loss_history, loss_grid.flatten()]))
        vmin = torch.min(torch.cat([loss_history, loss_grid.flatten()]))

        im = plt.imshow(loss_grid, cmap='Reds', vmax=vmax, vmin=vmin,
                        extent=[np.min(PC1_range), np.max(PC1_range),
                                np.max(PC2_range), np.min(PC2_range)])
        plt.colorbar(im)
        plt.scatter(PC1, PC2, c=loss_history, cmap='Reds', edgecolors='k', linewidths=0., vmax=vmax, vmin=vmin)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    else:
        fig = plt.figure()
        if network2 is None: # Set color scale
            vmax_net = vmax*torch.max(network1.val_loss_history)
        else:
            vmax_net = vmax*torch.max(torch.cat([network1.val_loss_history, network2.val_loss_history]))
        vmax_grid = torch.max(loss_grid)
        vmax = torch.min(vmax_grid,vmax_net)

        im = plt.imshow(loss_grid, cmap='Reds', vmax=vmax,
                        extent=[np.min(PC1_range), np.max(PC1_range),
                                np.max(PC2_range), np.min(PC2_range)])
        plt.colorbar(im)
        plt.scatter(PC1, PC2, s=0.1, color='k')

        if network2 is None:
            plt.scatter(PC1[0], PC2[0], s=80, color='b', edgecolor='k', label='Start')
            plt.scatter(PC1[-1], PC2[-1], s=80, color='orange', edgecolor='k', label='End')

        else:
            plt.scatter(PC1_network1[0], PC2_network1[0], s=80, color='b', edgecolor='k', label='Start')
            plt.scatter(PC1_network2[0], PC2_network2[0], s=80, color='b', edgecolor='k')

            if not hasattr(network1, 'name'):
                network1.name = '1'
            if not hasattr(network2, 'name'):
                network2.name = '2'
            plt.scatter(PC1_network1[-1], PC2_network1[-1], s=80, color='orange', edgecolor='k',
                        label=f'End {network1.name}')
            plt.scatter(PC1_network2[-1], PC2_network2[-1], s=80, color='cyan', edgecolor='k',
                        label=f'End {network2.name}')

        plt.legend()
        plt.xlabel(f'PC1 \n({percent_exp_var[0]}% var. explained)')
        plt.ylabel(f'PC2 \n({percent_exp_var[1]}% var. explained)')

    return fig


def plot_loss_landscape_multiple(test_network, param_history_dict, test_dataloader, num_points=20, extension=0.2):
    """
    Plot PCA loss landscape for combined weights from multiple networks
    """

    flat_param_history_all = []

    for network_name in param_history_dict:
        for i in param_history_dict[network_name]:
            flat_param_history, param_metadata = get_flat_param_history(param_history_dict[network_name][i])
            flat_param_history_all.append(flat_param_history)
    flat_param_history_all = torch.cat(flat_param_history_all)

    history_len = flat_param_history.shape[0]
    num_networks = ut.count_dict_elements(param_history_dict)
    flat_param_history = flat_param_history_all

    # Center the data (mean=0, std=1)
    p_mean = torch.mean(flat_param_history, axis=0)
    p_std = torch.std(flat_param_history, axis=0)
    centered_param_history = (flat_param_history - p_mean) / (p_std + 1e-10)  # add epsilon to avoid NaNs

    # Get weights in gridplane defined by PC dimensions
    pca = PCA(n_components=2)
    pca.fit(centered_param_history)
    param_hist_pca_space = pca.transform(centered_param_history)

    PC1 = param_hist_pca_space[:, 0]
    PC2 = param_hist_pca_space[:, 1]
    PC1_extension = (np.max(PC1) - np.min(PC1)) * extension
    PC2_extension = (np.max(PC2) - np.min(PC2)) * extension
    PC1_range = np.linspace(np.min(PC1) - PC1_extension, np.max(PC1) + PC1_extension, num_points)
    PC2_range = np.linspace(np.min(PC2) - PC2_extension, np.max(PC2) + PC2_extension, num_points)
    PC1_mesh, PC2_mesh = np.meshgrid(PC1_range, PC2_range)

    # Convert PC coordinates into full weight vectors
    flat_PC1_vals = PC1_mesh.reshape(1, num_points ** 2)
    flat_PC2_vals = PC2_mesh.reshape(1, num_points ** 2)
    meshgrid_points = np.concatenate([flat_PC1_vals, flat_PC2_vals]).T

    gridpoints_paramspace = pca.inverse_transform(meshgrid_points)
    gridpoints_paramspace = torch.tensor(gridpoints_paramspace) * p_std + p_mean

    # Compute loss for points in grid
    test_network = copy(test_network)  # create copy to avoid modifying original networks
    losses = torch.zeros(num_points ** 2)
    for i, gridpoint_flat in enumerate(tqdm(gridpoints_paramspace)):
        state_dict = unflatten_params(gridpoint_flat, param_metadata)
        losses[i] = compute_loss(test_network, state_dict, test_dataloader)
    loss_grid = losses.reshape([PC1_range.size, PC2_range.size])

    plot_loss_surface(loss_grid, PC1_mesh, PC2_mesh)

    lines_PC1 = []
    lines_PC2 = []
    for i in range(0, len(PC1), history_len):
        lines_PC1.append(PC1[i:i + history_len])
        lines_PC2.append(PC2[i:i + history_len])

    fig = plt.figure()
    im = plt.imshow(loss_grid, cmap='Reds',
                    extent=[np.min(PC1_range), np.max(PC1_range),
                            np.max(PC2_range), np.min(PC2_range)])
    plt.colorbar(im)

    plt.scatter(PC1, PC2, s=0.1, color='k')

    # plt.scatter(PC1_network1[0], PC2_network1[0], s=80, color='b', edgecolor='k', label='Start')
    # plt.scatter(PC1_network2[0], PC2_network2[0], s=80, color='b', edgecolor='k')

    # plt.scatter(PC1_network1[-1], PC2_network1[-1], s=80, color='orange', edgecolor='k',
    #             label=f'End {network1.name}')
    # plt.scatter(PC1_network2[-1], PC2_network2[-1], s=80, color='cyan', edgecolor='k',
    #             label=f'End {network2.name}')

    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')


def plot_3D_loss_surface(loss_grid, PC1_mesh, PC2_mesh):
    """
    Function plots loss surface from the grid based on PCs

    Parameters
    loss_grid: torch tensor, size: [num_points x num_points]
        values of loss for the given model at a given set of weights
    PC1_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape] (?)
    PC2_mesh: torch tensor, size: [1 x num_points(specified in get_loss_landscape]
    """
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(PC1_mesh, PC2_mesh, loss_grid.numpy(), cmap='terrain', alpha=0.9)

    ax.view_init(elev=30, azim=-50)
    ax.set_xlabel(r'PC1', labelpad=9)
    ax.set_ylabel(r'PC2', labelpad=10)
    ax.set_zlabel(r'Loss', labelpad=25)
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    ax.tick_params(axis='z', pad=10)
    plt.tight_layout()

    return fig


def plot_FB_weight_alignment(*projections, title=None):
    """

    :param projections: list of :class:'Projection'
    :param labels: list of str
    """
    fig, axes = plt.subplots()
    indexes = torch.where((projections[1].weight.data.T.flatten() == projections[1].initial_weight.T.flatten())
                          & (projections[0].weight.data.flatten() == projections[0].initial_weight.flatten()))
    axes.scatter(projections[0].weight.data.flatten()[indexes], projections[1].weight.data.T.flatten()[indexes], s=4.,
                 label='Un-allocated weights', color='orange', zorder=3)
    axes.scatter(projections[0].weight.data.flatten(), projections[1].weight.data.T.flatten(), s=4.,
                 label='Learned weights', color='b', zorder=2)
    axes.set_xlabel('Bottom-up weights')
    axes.set_ylabel('Top-down weights')
    if title is not None:
        axes.set_title(title)
    axes.legend(loc='best', frameon=False)
    fig.tight_layout()
    clean_axes(axes)
    fig.show()
