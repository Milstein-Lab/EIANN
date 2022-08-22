import torch
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def n_choose_k(n,k):
    num_permutations = np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    return int(num_permutations)


def n_hot_patterns(n,length):
    all_permutations = torch.tensor(list(itertools.product([0., 1.], repeat=length)))
    pattern_hotness = torch.sum(all_permutations,axis=1)
    idx = torch.where(pattern_hotness == n)[0]
    n_hot_patterns = all_permutations[idx]
    return n_hot_patterns


def get_diag_argmax_row_indexes(data):
    """
    Sort the rows of a square matrix such that whenever row argmax and col argmax are equal, that value appears
    on the diagonal. Returns row indexes.
    :param data: 2d array; square matrix
    :return: array of int
    """
    data = np.array(data)
    if data.shape[0] != data.shape[1]:
        raise Exception('get_diag_argmax_row_indexes: data must be a square matrix')
    dim = data.shape[0]
    avail_row_indexes = list(range(dim))
    avail_col_indexes = list(range(dim))
    final_row_indexes = np.empty_like(avail_row_indexes)
    while(len(avail_col_indexes) > 0):
        row_selectivity = np.zeros_like(avail_row_indexes)
        row_max = np.max(data[avail_row_indexes, :][:, avail_col_indexes], axis=1)
        row_mean = np.mean(data[avail_row_indexes,:][:,avail_col_indexes], axis=1)
        nonzero_indexes = np.where(row_mean > 0)
        row_selectivity[nonzero_indexes] = row_max[nonzero_indexes] / row_mean[nonzero_indexes]

        row_index = avail_row_indexes[np.argsort(row_selectivity)[-1]]
        col_index = avail_col_indexes[np.argmax(data[row_index,avail_col_indexes])]
        final_row_indexes[col_index] = row_index
        avail_row_indexes.remove(row_index)
        avail_col_indexes.remove(col_index)
    return final_row_indexes


def test_EIANN_config(network, dataset, target, epochs):

    for sample in dataset:
        network.forward(sample, store_history=True)

    reversed_layers = list(network)
    reversed_layers.reverse()
    output_pop = next(iter(reversed_layers[0]))

    max_rows = 1
    cols = len(network.layers) - 1
    for layer in network:
        projection_count = 0
        for population in layer:
            projection_count += len(list(population))
        max_rows = max(max_rows, projection_count)

    fig, axes = plt.subplots(max_rows, cols, figsize=(3.*cols, 3.*max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            row = 0
            for population in layer:
                for projection in population:
                    if cols == 1:
                        if max_rows == 1:
                            this_axis = axes
                        else:
                            this_axis = axes[row]
                    elif max_rows == 1:
                        this_axis = axes[col]
                    else:
                        this_axis = axes[row, col]
                    im = this_axis.imshow(projection.weight.data, aspect='auto')
                    fig.colorbar(im, ax=this_axis)
                    this_axis.set_xlabel('Pre unit ID')
                    this_axis.set_ylabel('Post unit ID')
                    this_axis.set_title('%s.%s <- %s.%s' %
                              (projection.post.layer.name, projection.post.name,
                               projection.pre.layer.name, projection.pre.name))
                    row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Initial weights')
    fig.tight_layout()
    fig.show()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for col, layer in enumerate(network):
        for row, population in enumerate(layer):
            if cols == 1:
                if max_rows == 1:
                    this_axis = axes
                else:
                    this_axis = axes[row]
            elif max_rows == 1:
                this_axis = axes[col]
            else:
                this_axis = axes[row, col]
            im = this_axis.imshow(population.activity_history[-dataset.shape[0]:, -1, :].T, aspect='auto')
            fig.colorbar(im, ax=this_axis)
            this_axis.set_xlabel('Input pattern ID')
            this_axis.set_ylabel('Output unit ID')
            this_axis.set_title('%s.%s' % (layer.name, population.name))
        row += 1
        while row < max_rows:
            if cols == 1:
                this_axis = axes[row]
            else:
                this_axis = axes[row, col]
            this_axis.set_visible(False)
            row += 1
    fig.suptitle('Initial activity')
    fig.tight_layout()
    fig.show()

    cols = len(network.layers) - 1
    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                if cols == 1:
                    if max_rows == 1:
                        this_axis = axes
                    else:
                        this_axis = axes[row]
                elif max_rows == 1:
                    this_axis = axes[col]
                else:
                    this_axis = axes[row, col]
                for i in range(population.size):
                    this_axis.plot(torch.mean(population.activity_history[-dataset.shape[0]:, :, i], axis=0))
                this_axis.set_xlabel('Equilibration time steps')
                this_axis.set_ylabel('Unit ID')
                this_axis.set_title('%s.%s' % (layer.name, population.name))
            row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Mean activity dynamics')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)

    network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=True, status_bar=True)

    final_output = output_pop.activity_history[network.sorted_sample_indexes, -1, :][-dataset.shape[0]:, :].T
    if final_output.shape[0] == final_output.shape[1]:
        sorted_idx = get_diag_argmax_row_indexes(final_output)
    else:
        sorted_idx = np.arange(final_output.shape[0])

    sorted_loss_history = []
    for i in range(output_pop.activity_history.shape[0]):
        sample_idx = network.sample_order[i]
        sample_target = target[sample_idx,:]
        output = output_pop.activity_history[i,-1, sorted_idx]
        loss = network.criterion(output, sample_target)
        sorted_loss_history.append(loss)
    sorted_loss_history = torch.tensor(sorted_loss_history)

    fig = plt.figure()
    plt.plot(network.loss_history, label='Unsorted')
    plt.plot(sorted_loss_history, label='Sorted')
    plt.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', frameon=False)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Training loss')
    fig.tight_layout()
    fig.show()

    max_rows = 1
    cols = len(network.layers) - 1
    for layer in network:
        projection_count = 0
        for population in layer:
            projection_count += len(list(population))
        max_rows = max(max_rows, projection_count)

    fig, axes = plt.subplots(max_rows, cols, figsize=(3.*cols, 3.*max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            row = 0
            for population in layer:
                for projection in population:
                    if cols == 1:
                        if max_rows == 1:
                            this_axis = axes
                        else:
                            this_axis = axes[row]
                    elif max_rows == 1:
                        this_axis = axes[col]
                    else:
                        this_axis = axes[row, col]
                    if projection.post == output_pop:
                        im = this_axis.imshow(projection.weight.data[sorted_idx,:], aspect='auto')
                    else:
                        im = this_axis.imshow(projection.weight.data, aspect='auto')
                    fig.colorbar(im, ax=this_axis)
                    this_axis.set_xlabel('Pre unit ID')
                    this_axis.set_ylabel('Post unit ID')
                    this_axis.set_title('%s.%s <- %s.%s' %
                              (projection.post.layer.name, projection.post.name,
                               projection.pre.layer.name, projection.pre.name))
                    row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Final weights')
    fig.tight_layout()
    fig.show()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for col, layer in enumerate(network):
        for row, population in enumerate(layer):
            if cols == 1:
                if max_rows == 1:
                    this_axis = axes
                else:
                    this_axis = axes[row]
            elif max_rows == 1:
                this_axis = axes[col]
            else:
                this_axis = axes[row, col]
            if population == output_pop:
                im = this_axis.imshow(population.activity_history[network.sorted_sample_indexes, -1, :][
                                      -dataset.shape[0]:, sorted_idx].T, aspect='auto')
            else:
                im = this_axis.imshow(population.activity_history[network.sorted_sample_indexes, -1, :][
                                      -dataset.shape[0]:, :].T, aspect='auto')
            fig.colorbar(im, ax=this_axis)
            this_axis.set_xlabel('Input pattern ID')
            this_axis.set_ylabel('Output unit ID')
            this_axis.set_title('%s.%s' % (layer.name, population.name))
        row += 1
        while row < max_rows:
            if cols == 1:
                this_axis = axes[row]
            else:
                this_axis = axes[row, col]
            this_axis.set_visible(False)
            row += 1
    fig.suptitle('Final activity')
    fig.tight_layout()
    fig.show()

    cols = len(network.layers) - 1
    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                if cols == 1:
                    if max_rows == 1:
                        this_axis = axes
                    else:
                        this_axis = axes[row]
                elif max_rows == 1:
                    this_axis = axes[col]
                else:
                    this_axis = axes[row, col]
                for i in range(population.size):
                    this_axis.plot(torch.mean(population.activity_history[-dataset.shape[0]:, :, i], axis=0))
                this_axis.set_xlabel('Equilibration time steps')
                this_axis.set_ylabel('Unit ID')
                this_axis.set_title('%s.%s' % (layer.name, population.name))
            row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Mean activity dynamics')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, 'bias:', population.bias)


def test_EIANN_config2(network, dataset, target, epochs):

    for sample in dataset:
        network.forward(sample, store_history=True)

    reversed_layers = list(network)
    reversed_layers.reverse()
    output_pop = next(iter(reversed_layers[0]))

    max_rows = 1
    cols = len(network.layers) - 1
    for layer in network:
        projection_count = 0
        for population in layer:
            projection_count += len(list(population))
        max_rows = max(max_rows, projection_count)

    fig, axes = plt.subplots(max_rows, cols, figsize=(3.*cols, 3.*max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            row = 0
            for population in layer:
                for projection in population:
                    if cols == 1:
                        if max_rows == 1:
                            this_axis = axes
                        else:
                            this_axis = axes[row]
                    elif max_rows == 1:
                        this_axis = axes[col]
                    else:
                        this_axis = axes[row, col]
                    im = this_axis.imshow(projection.weight.data, aspect='auto')
                    fig.colorbar(im, ax=this_axis)
                    this_axis.set_xlabel('Pre unit ID')
                    this_axis.set_ylabel('Post unit ID')
                    this_axis.set_title('%s.%s <- %s.%s' %
                              (projection.post.layer.name, projection.post.name,
                               projection.pre.layer.name, projection.pre.name))
                    row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Initial weights')
    fig.tight_layout()
    fig.show()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for col, layer in enumerate(network):
        for row, population in enumerate(layer):
            if cols == 1:
                if max_rows == 1:
                    this_axis = axes
                else:
                    this_axis = axes[row]
            elif max_rows == 1:
                this_axis = axes[col]
            else:
                this_axis = axes[row, col]
            im = this_axis.imshow(population.activity_history[-dataset.shape[0]:, -1, :].T, aspect='auto')
            fig.colorbar(im, ax=this_axis)
            this_axis.set_xlabel('Input pattern ID')
            this_axis.set_ylabel('Output unit ID')
            this_axis.set_title('%s.%s' % (layer.name, population.name))
        row += 1
        while row < max_rows:
            if cols == 1:
                this_axis = axes[row]
            else:
                this_axis = axes[row, col]
            this_axis.set_visible(False)
            row += 1
    fig.suptitle('Initial activity')
    fig.tight_layout()
    fig.show()

    cols = len(network.layers) - 1
    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                if cols == 1:
                    if max_rows == 1:
                        this_axis = axes
                    else:
                        this_axis = axes[row]
                elif max_rows == 1:
                    this_axis = axes[col]
                else:
                    this_axis = axes[row, col]
                for i in range(population.size):
                    this_axis.plot(torch.mean(population.activity_history[-dataset.shape[0]:, :, i], axis=0))
                this_axis.set_xlabel('Equilibration time steps')
                this_axis.set_ylabel('Unit ID')
                this_axis.set_title('%s.%s' % (layer.name, population.name))
            row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Mean activity dynamics')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)

    network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=False, status_bar=True)

    network.reset_history()

    for sample in dataset:
        network.forward(sample, store_history=True)

    final_output = output_pop.activity_history[network.sorted_sample_indexes, -1, :][-dataset.shape[0]:, :].T
    """
    if final_output.shape[0] == final_output.shape[1]:
        sorted_idx = get_diag_argmax_row_indexes(final_output)
    else:
    """
    sorted_idx = np.arange(final_output.shape[0])

    """
    sorted_loss_history = []
    for i in range(output_pop.activity_history.shape[0]):
        sample_idx = network.sample_order[i]
        sample_target = target[sample_idx,:]
        output = output_pop.activity_history[i,-1, sorted_idx]
        loss = network.criterion(output, sample_target)
        sorted_loss_history.append(loss)
    sorted_loss_history = torch.tensor(sorted_loss_history)

    fig = plt.figure()
    plt.plot(network.loss_history, label='Unsorted')
    plt.plot(sorted_loss_history, label='Sorted')
    plt.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', frameon=False)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Training loss')
    fig.tight_layout()
    fig.show()
    """

    max_rows = 1
    cols = len(network.layers) - 1
    for layer in network:
        projection_count = 0
        for population in layer:
            projection_count += len(list(population))
        max_rows = max(max_rows, projection_count)

    fig, axes = plt.subplots(max_rows, cols, figsize=(3.*cols, 3.*max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            row = 0
            for population in layer:
                for projection in population:
                    if cols == 1:
                        if max_rows == 1:
                            this_axis = axes
                        else:
                            this_axis = axes[row]
                    elif max_rows == 1:
                        this_axis = axes[col]
                    else:
                        this_axis = axes[row, col]
                    if projection.post == output_pop:
                        im = this_axis.imshow(projection.weight.data[sorted_idx,:], aspect='auto')
                    else:
                        im = this_axis.imshow(projection.weight.data, aspect='auto')
                    fig.colorbar(im, ax=this_axis)
                    this_axis.set_xlabel('Pre unit ID')
                    this_axis.set_ylabel('Post unit ID')
                    this_axis.set_title('%s.%s <- %s.%s' %
                              (projection.post.layer.name, projection.post.name,
                               projection.pre.layer.name, projection.pre.name))
                    row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Final weights')
    fig.tight_layout()
    fig.show()

    max_rows = 1
    cols = len(network.layers)
    for layer in network:
        max_rows = max(max_rows, len(layer.populations))

    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for col, layer in enumerate(network):
        for row, population in enumerate(layer):
            if cols == 1:
                if max_rows == 1:
                    this_axis = axes
                else:
                    this_axis = axes[row]
            elif max_rows == 1:
                this_axis = axes[col]
            else:
                this_axis = axes[row, col]
            if population == output_pop:
                im = this_axis.imshow(population.activity_history[network.sorted_sample_indexes, -1, :][
                                      -dataset.shape[0]:, sorted_idx].T, aspect='auto')
            else:
                im = this_axis.imshow(population.activity_history[network.sorted_sample_indexes, -1, :][
                                      -dataset.shape[0]:, :].T, aspect='auto')
            fig.colorbar(im, ax=this_axis)
            this_axis.set_xlabel('Input pattern ID')
            this_axis.set_ylabel('Output unit ID')
            this_axis.set_title('%s.%s' % (layer.name, population.name))
        row += 1
        while row < max_rows:
            if cols == 1:
                this_axis = axes[row]
            else:
                this_axis = axes[row, col]
            this_axis.set_visible(False)
            row += 1
    fig.suptitle('Final activity')
    fig.tight_layout()
    fig.show()

    cols = len(network.layers) - 1
    fig, axes = plt.subplots(max_rows, cols, figsize=(3. * cols, 3. * max_rows))
    for i, layer in enumerate(network):
        if i > 0:
            col = i - 1
            for row, population in enumerate(layer):
                if cols == 1:
                    if max_rows == 1:
                        this_axis = axes
                    else:
                        this_axis = axes[row]
                elif max_rows == 1:
                    this_axis = axes[col]
                else:
                    this_axis = axes[row, col]
                for i in range(population.size):
                    this_axis.plot(torch.mean(population.activity_history[-dataset.shape[0]:, :, i], axis=0))
                this_axis.set_xlabel('Equilibration time steps')
                this_axis.set_ylabel('Unit ID')
                this_axis.set_title('%s.%s' % (layer.name, population.name))
            row += 1
            while row < max_rows:
                if cols == 1:
                    this_axis = axes[row]
                else:
                    this_axis = axes[row, col]
                this_axis.set_visible(False)
                row += 1
    fig.suptitle('Mean activity dynamics')
    fig.tight_layout()
    fig.show()

    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, 'bias:', population.bias)