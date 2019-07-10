import logging

logging.captureWarnings(True)
log = logging.getLogger("spiking-mnist")

import os.path
import numpy as np
from scipy import sparse
from scipy.special import betaincinv
import brian2 as b2
from urllib.request import urlretrieve

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits import axes_grid1
import matplotlib.patheffects as path_effects

import pandas as pd

from IPython import embed


def get_labeled_data():
    log.info("Loading MNIST data")
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    filename = "mnist.npz"
    localfilename = os.path.join("data", filename)
    if not os.path.exists(localfilename):
        os.makedirs("data", exist_ok=True)
        urlretrieve(origin_folder + filename, localfilename)
    with np.load(localfilename) as f:
        training = {"x": f["x_train"], "y": f["y_train"]}
        testing = {"x": f["x_test"], "y": f["y_test"]}
    return training, testing


def get_matrix_from_file(filename, shape=None):
    log.debug(f"Reading matrix from {filename}")
    i, j, data = np.load(filename).T
    i = i.astype(np.int)
    j = j.astype(np.int)
    log.debug(f"Read {len(data)} connections")
    arr = sparse.coo_matrix((data, (i, j)), shape).todense()
    log.debug(f"Created a matrix with shape {arr.shape}")
    # log.debug("Statistics:\n" + pd.Series(arr.flat).describe().to_string())
    return arr


def connections_to_file(conn, filename):
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save(filename, connListSparse)


def get_initial_weights(n_input, n_e):
    matrices = {}
    npr = np.random.RandomState(9728364)
    matrices['AeAi'] = np.eye(n_e) * 10.4
    matrices['AiAe'] = 17.0 * (1 - np.eye(n_e))
    matrices['XeAe'] = npr.uniform(0.003, 0.303, (n_input, n_e))
    new = np.zeros((n_input, n_e))
    n_connect = int(0.1 * n_input * n_e)
    connect = npr.choice(n_input * n_e, n_connect, replace=False)
    new.flat[connect] = npr.uniform(0.0, 0.2, n_connect)
    matrices['XeAi'] = new
    return matrices


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
            rearranged_weights[
                i * n_in_sqrt : (i + 1) * n_in_sqrt, j * n_in_sqrt : (j + 1) * n_in_sqrt
            ] = wk
    return rearranged_weights


def plot_weights(weights, assignments=None, max_weight=1.0, ax=None, filename=None):
    if isinstance(weights, b2.Synapses):
        weights = sparse.coo_matrix((weights.w, (weights.i, weights.j))).todense()
    rearranged_weights = rearrange_weights(weights)
    closefig = False
    if ax is None:
        fig, ax = plt.subplots()
        closefig = True
    im = ax.imshow(
        rearranged_weights,
        interpolation="nearest",
        vmin=0,
        vmax=max_weight,
        cmap=cm.hot_r,
    )
    add_colorbar(im)
    if assignments is not None:
        n_in_sqrt = int(np.sqrt(weights.shape[0]))
        n_e = weights.shape[1]
        n_e_sqrt = int(np.sqrt(n_e))
        a = np.zeros(n_e, np.int) - 1
        a[assignments.index] = assignments["label"]
        a = a.reshape((n_e_sqrt, n_e_sqrt))
        a = a.astype(np.str)
        a[a == "-1"] = ""
        for i in range(n_e_sqrt):
            for j in range(n_e_sqrt):
                txt = ax.text(
                    (1 + i) * n_in_sqrt - 3,
                    (1 + j) * n_in_sqrt - 3,
                    a[i, j],
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    fontsize=4,
                )
                txt.set_path_effects(
                    [path_effects.withStroke(linewidth=1, foreground="w")]
                )
    if filename is not None:
        ax.get_figure().savefig(filename)
    if closefig:
        plt.close(fig)


def plot_accuracy(acchist, ax=None, filename=None):
    closefig = False
    if ax is None:
        fig, ax = plt.subplots()
        closefig = True
    i = np.array(list(acchist.keys()))
    a = np.array(list(acchist.values()))
    amid, alow, ahigh = a.T
    ax.fill_between(i, alow, ahigh, facecolor="blue", alpha=0.5)
    ax.plot(i, amid, color="blue")
    ax.set_xlabel("examples seen")
    ax.set_ylabel("accuracy")
    ax.set_xlim(xmin=0)
    if filename is not None:
        ax.get_figure().savefig(filename)
    if closefig:
        plt.close(fig)


def spike_counts_from_cumulative(
    cumulative_spike_counts, n_data, start=0, end=None, atmost=None
):
    log.debug("Producing spike counts from cumulative counts")
    counts = np.diff(cumulative_spike_counts, axis=0)
    ntbin = len(counts)
    log.debug("Counts for {} examples".format(ntbin))
    s = sparse.coo_matrix(counts)
    spikecounts = pd.DataFrame({"tbin": s.row, "i": s.col, "count": s.data})
    if start is not None and start < 0:
        start += ntbin
    if end is None:
        end = ntbin
    elif end < 0:
        end += ntbin
    if atmost is not None:
        if end is None:
            end = min(end, start + atmost)
        else:
            start = max(start, end - atmost)
    log.debug("Starting at tbin {}".format(start))
    spikecounts = spikecounts[spikecounts["tbin"] >= start]
    log.debug("Ending before tbin {}".format(end))
    spikecounts = spikecounts[spikecounts["tbin"] < end]
    spikecounts["example_idx"] = spikecounts["tbin"] % n_data
    spikecounts = spikecounts.set_index(["tbin", "i"])
    return spikecounts


def get_assignments(counts, labels):
    counts = counts.reset_index('i').set_index('example_idx')
    counts = labels.join(counts, how="right")
    counts = counts.reset_index("example_idx", drop=True)
    counts = counts.groupby(["i", "label"]).sum().reset_index("label")
    counts = counts.sort_values(["i", "count"], ascending=[True, False])
    assignments = counts.groupby("i").head(1).drop(columns="count")
    return assignments


def get_predictions(counts, assignments, labels=None):
    counts = pd.DataFrame(assignments).join(counts)
    counts = counts.rename(columns={"label": "assignment"})
    counts = counts.groupby(["tbin", "assignment"]).mean()
    counts = counts.sort_values(["tbin", "count"], ascending=[True, False])
    predictions = counts.groupby(["tbin"]).head(1)
    predictions = predictions.reset_index("assignment")
    if labels is not None:
        predictions = predictions.join(labels, on="example_idx")
    return predictions


def get_accuracy(predictions):
    match = predictions["assignment"] == predictions["label"]
    k = match.sum()
    n = len(predictions)
    if n == 0:
        return None
    mid = k / n
    lower, upper = binom_conf_interval(k, n)
    return np.array([mid, lower, upper])


def get_labels(data):
    return pd.DataFrame({"label": data["y"]}).rename_axis("example_idx")


# adapted from astropy.stats.funcs
def binom_conf_interval(k, n, conf=0.68269):
    """Binomial proportion confidence interval given k successes,
    n trials, adopting Bayesian approach with Jeffreys prior."""

    if conf < 0.0 or conf > 1.0:
        raise ValueError("conf must be between 0. and 1.")
    alpha = 1.0 - conf

    k = np.asarray(k).astype(int)
    n = np.asarray(n).astype(int)

    if (n <= 0).any():
        log.warning("%(funcName)s: n must be positive")
        return 0, 0
    if (k < 0).any() or (k > n).any():
        log.warning("%(funcName)s: k must be in {0, 1, .., n}")
        return 0, 0

    lowerbound = betaincinv(k + 0.5, n - k + 0.5, 0.5 * alpha)
    upperbound = betaincinv(k + 0.5, n - k + 0.5, 1.0 - 0.5 * alpha)

    # Set lower or upper bound to k/n when k/n = 0 or 1
    # We have to treat the special case of k/n being scalars,
    # which is an ugly kludge
    if lowerbound.ndim == 0:
        if k == 0:
            lowerbound = 0.0
        elif k == n:
            upperbound = 1.0
    else:
        lowerbound[k == 0] = 0
        upperbound[k == n] = 1

    conf_interval = np.array([lowerbound, upperbound])

    return conf_interval


# from https://stackoverflow.com/a/33505522/1840212
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
