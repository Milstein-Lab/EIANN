# TODO: 
# Plot receptive fields: make sure all the inputs (e.g. num rows/columns, num units, ax list, etc)
# Migrate other tests into this directory

import pytest
import torch
import os

import EIANN.utils as ut
import EIANN.plot as pt
import EIANN._network as nt


def test_plot_batch_accuracy(network, dataloaders_mnist):
    train_dataloader, train_sub_dataloader, val_dataloader, test_dataloader, data_generator =  dataloaders_mnist
    pt.plot_batch_accuracy(network, test_dataloader, population='OutputE', sorted_output_idx=None, title=None, ax=None)
    pt.plot_batch_accuracy(network, test_dataloader, population='H1E', sorted_output_idx=None, title='Test', ax=None)

