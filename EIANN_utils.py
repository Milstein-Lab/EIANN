import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt


def n_choose_k(n, k):
    '''
    Calculates number of ways to choose k things out of n, using binomial coefficients

    :param n: number of things to choose from
    :type n: int
    :param k: number of things chosen
    :type k: int
    :return: int
    '''
    assert n>k, "k must be smaller than n"
    num_permutations = np.math.factorial(n) / (np.math.factorial(k)*np.math.factorial(n-k))
    return int(num_permutations)


def n_hot_patterns(n, length):
    '''
    Generates all possible binary n-hot patterns of given length

    :param n: number of bits set to 1
    :type n: int
    :param length: size of pattern (number of bits)
    :type length: int
    :return: torch.tensor
    '''
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


def plot_EIANN_activity(network, num_samples, supervised=True, label=None):

    reversed_layers = list(network)
    reversed_layers.reverse()
    output_pop = next(iter(reversed_layers[0]))

    if len(network.sorted_sample_indexes) > 0:
        sorted_sample_indexes = network.sorted_sample_indexes
    else:
        sorted_sample_indexes = torch.arange(0, output_pop.activity_history.shape[0])

    output = output_pop.activity_history[sorted_sample_indexes, -1, :][-num_samples:, :].T
    if not supervised and output.shape[0] == output.shape[1]:
        sorted_idx = get_diag_argmax_row_indexes(output)
    else:
        sorted_idx = np.arange(output.shape[0])

    if label is None:
        label_str = ''
    else:
        label_str = '%s ' % label

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
                        im = this_axis.imshow(projection.weight.data[sorted_idx, :], aspect='auto')
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
    fig.suptitle('%sweights' % label_str)
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
                im = this_axis.imshow(population.activity_history[sorted_sample_indexes, -1, :][
                                      -num_samples:, sorted_idx].T, aspect='auto')
            else:
                im = this_axis.imshow(population.activity_history[sorted_sample_indexes, -1, :][
                                      -num_samples:, :].T, aspect='auto')
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
    fig.suptitle('%sactivity' % label_str)
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
                    this_axis.plot(torch.mean(population.activity_history[-num_samples:, :, i], axis=0))
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
    fig.suptitle('%sactivity dynamics' % label_str)
    fig.tight_layout()
    fig.show()

    print('%spopulation biases:' % label_str)
    for i, layer in enumerate(network):
        if i > 0:
            for population in layer:
                print(layer.name, population.name, population.bias)


def analyze_EIANN_loss(network, target, supervised=True, plot=False):

    reversed_layers = list(network)
    reversed_layers.reverse()
    output_pop = next(iter(reversed_layers[0]))

    argmax_correct = []
    if supervised or output.shape[0] != output.shape[1]:
        loss_history = network.loss_history
        for i in range(output_pop.activity_history.shape[0]):
            sample_idx = network.sample_order[i]
            sample_target = target[sample_idx, :]
            output = output_pop.activity_history[i, -1, :]
            argmax_correct.append(torch.argmax(output) == torch.argmax(sample_target))
    else:
        final_output = output_pop.activity_history[network.sorted_sample_indexes, -1, :][-target.shape[0]:, :].T
        sorted_idx = get_diag_argmax_row_indexes(output)
        loss_history = []
        for i in range(output_pop.activity_history.shape[0]):
            sample_idx = network.sample_order[i]
            sample_target = target[sample_idx, :]
            output = output_pop.activity_history[i, -1, sorted_idx]
            loss = network.criterion(output, sample_target)
            loss_history.append(loss)
            argmax_correct.append(torch.argmax(output) == torch.argmax(sample_target))
        loss_history = torch.tensor(loss_history)
    argmax_correct = torch.tensor(argmax_correct)

    epoch_argmax_accuracy = []
    start = 0
    while start < len(argmax_correct):
        epoch_argmax_accuracy.append(torch.sum(argmax_correct[start:start+target.shape[0]]) / target.shape[0] * 100.)
        start += target.shape[0]

    epoch_argmax_accuracy = torch.tensor(epoch_argmax_accuracy)

    if plot:
        fig = plt.figure()
        plt.plot(loss_history)
        plt.xlabel('Training steps')
        plt.ylabel('MSE loss')
        plt.title('Training loss')
        fig.tight_layout()
        fig.show()

        fig = plt.figure()
        plt.plot(epoch_argmax_accuracy)
        plt.xlabel('Training epochs')
        plt.ylabel('% correct argmax')
        plt.title('Argmax accuracy')
        fig.tight_layout()
        fig.show()

    return loss_history, epoch_argmax_accuracy


def test_EIANN_config(network, dataset, target, epochs, supervised=True):

    for sample in dataset:
        network.forward(sample, store_history=True)
    plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=supervised, label='Initial')
    network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=True, status_bar=True)
    loss_history, epoch_argmax_accuracy = analyze_EIANN_loss(network, target, supervised=supervised, plot=True)
    plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=supervised, label='Final')
