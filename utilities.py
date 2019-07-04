import logging
logging.captureWarnings(True)
log = logging.getLogger('spiking-mnist')

import os.path
import numpy as np
from scipy import sparse
import brian2 as b2
from keras.datasets import mnist

from matplotlib import pyplot as plt
from matplotlib import cm

import pandas as pd


def get_labeled_data():
    log.info('Loading MNIST data')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    training = {'x': x_train, 'y': y_train}
    testing = {'x': x_test, 'y': y_test}
    return training, testing


def get_matrix_from_file(filename, shape=None):
    log.debug(f'Reading matrix from {filename}')
    i, j, data = np.load(filename).T
    i = i.astype(np.int)
    j = j.astype(np.int)
    log.debug(f'Read {len(data)} connections')
    arr = sparse.coo_matrix((data, (i, j)), shape).todense()
    log.debug(f'Created a matrix with shape {arr.shape}')
    return arr


def rearrange_weights(weights):
    n_input = weights.shape[0]
    n_e = weights.shape[1]
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    weights = np.reshape(weights, (n_input, n_e))
    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            wk = weights[:, i + j * n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
            rearranged_weights[i * n_in_sqrt: (i + 1) * n_in_sqrt,
                               j * n_in_sqrt: (j + 1) * n_in_sqrt] = wk
    return rearranged_weights


def plot_weights(weights, assignments=None, max_weight=1.0):
    rearranged_weights = rearrange_weights(weights)
    fig, ax = plt.subplots(figsize=(18, 18))
    im = ax.imshow(rearranged_weights, interpolation="nearest", vmin=0,
                   vmax=max_weight, cmap=cm.hot_r)
    #plt.colorbar(im)
    if assignments is not None:
        n_in_sqrt = int(np.sqrt(weights.shape[0]))
        n_e_sqrt = int(np.sqrt(weights.shape[1]))
        a = assignments.values.reshape((n_e_sqrt, n_e_sqrt))
        for i in range(n_e_sqrt):
            for j in range(n_e_sqrt):
                ax.text((1 + i) * n_in_sqrt - 3,
                        (1 + j) * n_in_sqrt - 3,
                        a[i, j],
                        horizontalalignment='right',
                        verticalalignment='bottom')

def spike_counts_from_cumulative(cumulative_spike_counts):
    counts = np.diff(cumulative_spike_counts, axis=0)
    s = sparse.coo_matrix(counts)
    spikecounts = pd.DataFrame({'tbin': s.row, 'i': s.col, 'count': s.data})
    spikecounts = spikecounts.set_index(['tbin', 'i'])
    return spikecounts
