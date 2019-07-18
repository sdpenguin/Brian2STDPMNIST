# Functions for post-run analysis
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import brian2 as b2


def process_spike_trains(monitors, total_example_time, imgsize=28):
    spikes = {}
    for p in monitors:
        # convert time to integers in units of 0.1 ms
        spikes[p] = pd.DataFrame({'t': (monitors[p]['t'] / (0.1 * b2.ms)).astype(np.int),
                                  'i'.format(p): monitors[p]['i']})
        spikes[p]['tbin'] = (spikes[p]['t'] * 0.1 * b2.ms / total_example_time).astype(np.int)
        if 'X' in p:
            spikes[p]['x'] = spikes[p]['i'] % imgsize
            spikes[p]['y'] = spikes[p]['i'] // imgsize
    return spikes


def bin_spike_trains(spikes, n_data):
    spikecounts = {}
    for p in spikes:
        counts = spikes[p].groupby(["tbin", 'i'])['t'].count().rename('count')
        counts = pd.DataFrame(counts).reset_index()
        counts["example_idx"] = counts["tbin"] % n_data
        spikecounts[p] = counts.set_index(["tbin", "i"])
    return spikecounts


def plot_spike_distribution(counts, ax=None):
    closefig = False
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig = None
        ax1, ax2 = ax
    counts_mean = counts['count'].groupby('i').mean()
    counts_std = counts['count'].groupby('i').std()
    ax1.hist(counts_mean)
    ax1.set_xlabel('spikes per example')
    ax2.errorbar(counts_mean.index, counts_mean, counts_std, marker='', linestyle='none')
    ax2.set_xlabel('neuron index')
    ax2.set_ylabel('spikes per example')
    if closefig:
        plt.close(fig)
    return fig


def plot_assignment_distribution(assignments, labels, ax=None):
    closefig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        closefig = True
    else:
        fig = None
    n, b, p = ax.hist(labels['label'], density=True, bins=10, range=[-0.5, 9.5])
    ax.hist(assignments['label'], density=True, bins=b, histtype='step', lw=2)
    ax.set_xlabel('label')
    ax.xaxis.set_ticks(np.unique(labels))
    if closefig:
        plt.close(fig)
    return fig
