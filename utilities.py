import logging

logging.captureWarnings(True)
log = logging.getLogger("spiking-mnist")

import os.path
from math import ceil
import numpy as np
from scipy import sparse
from scipy.special import betaincinv
import brian2 as b2
from urllib.request import urlretrieve
from inspect import getargvalues

import matplotlib

matplotlib.use("PDF")
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


def connections_to_pandas(conn, nseen):
    df = pd.DataFrame(
        {
            "i": conn.i[:].astype(np.int32),
            "j": conn.j[:].astype(np.int32),
            "w": conn.w[:].astype(np.float32),
        }
    )
    df["nseen"] = nseen
    df = df.set_index("nseen", append=True)
    return df


def theta_to_pandas(subpop, neuron_groups, nseen):
    t = pd.Series(neuron_groups[subpop].theta[:] / b2.mV, dtype=np.float32)
    t = add_nseen_index(t, nseen)
    return t


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
    return rearranged_weights, n_in_sqrt, n_e_sqrt


def rearrange_output_weights(weights):
    n_e = weights.shape[0]
    n_output = weights.shape[1]
    n_e_sqrt = int(np.sqrt(n_e))
    num_values_col = n_e_sqrt
    num_values_row = n_e_sqrt * n_output
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    for i in range(n_output):
        wk = weights[:, i].reshape((n_e_sqrt, n_e_sqrt)).T
        rearranged_weights[:, i * n_e_sqrt : (i + 1) * n_e_sqrt] = wk
    return rearranged_weights, n_e_sqrt, n_output


def plot_weights(
    weights,
    assignments=None,
    theta=None,
    max_weight=1.0,
    ax=None,
    filename=None,
    return_artists=False,
    nseen=None,
    output=False,
    feedback=False,
    label="",
):
    log.debug(f"Plotting weights {label}")
    log.debug(f"output={output}, feedback={feedback}")
    # log.debug(f"weights = \n{weights}")
    if isinstance(weights, b2.Synapses):
        weights = sparse.coo_matrix((weights.w, (weights.i, weights.j))).todense()
    # log.debug(f"dense weights shape = {weights.shape}")
    if output:
        if feedback:
            weights = weights.T
        rearranged_weights, n, m = rearrange_output_weights(weights)
        figsize = (8, 3)
    else:
        rearranged_weights, n, m = rearrange_weights(weights)
        figsize = (8, 7)
    # log.debug(f"rearranged weights shape = {rearranged_weights.shape}")
    fig, ax, closefig = openfig(ax, figsize=figsize)
    if max_weight is None:
        max_weight = rearranged_weights.max() * 1.1
        if not output and max_weight > 0.1:
            # quantize to 0.25, 0.50, 0.75, 1.00
            max_weight = ceil(max_weight * 4) / 4
    im = ax.imshow(
        rearranged_weights,
        interpolation="nearest",
        vmin=0,
        vmax=max_weight,
        cmap=cm.plasma,
    )
    if not output:
        plt.hlines(np.arange(1, m) * n - 0.5, -0.5, n * m - 0.5, lw=0.5, colors="w")
        plt.vlines(np.arange(1, m) * n - 0.5, -0.5, n * m - 0.5, lw=0.5, colors="w")
    else:
        plt.vlines(np.arange(1, m) * n - 0.5, -0.5, n - 0.5, lw=0.5, colors="w")
    ax.yaxis.set_ticks([])
    if nseen is not None:
        ax.set_title(f"examples seen: {nseen: 6d}", loc="right")
    if output:
        ax.xaxis.set_ticks(np.arange(m) * n + n // 2)
        ax.xaxis.set_ticklabels(np.arange(m))
        cbar_aspect = 5
    else:
        ax.xaxis.set_ticks([])
        cbar_aspect = 20
    cbar = add_colorbar(im, aspect=cbar_aspect)
    cbar.set_label(f"weight {label}")
    theta_text = []
    assignments_text = []
    if assignments is not None or theta is not None:
        n_in_sqrt = int(np.sqrt(weights.shape[0]))
        n_e = weights.shape[1]
        n_e_sqrt = int(np.sqrt(n_e))
        if assignments is not None:
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
                        fontsize="x-small",
                    )
                    txt.set_path_effects(
                        [path_effects.withStroke(linewidth=1, foreground="w")]
                    )
                    assignments_text.append(txt)
        if theta is not None:
            t = theta.values.reshape((n_e_sqrt, n_e_sqrt)) * 1000  # mV
            for i in range(n_e_sqrt):
                for j in range(n_e_sqrt):
                    txt = ax.text(
                        (1 + i) * n_in_sqrt - 3,
                        j * n_in_sqrt + 3,
                        f"{t[i, j]:4.2f}",
                        horizontalalignment="right",
                        verticalalignment="top",
                        fontsize="x-small",
                    )
                    txt.set_path_effects(
                        [path_effects.withStroke(linewidth=1, foreground="w")]
                    )
                    theta_text.append(txt)
    endfig(filename, fig, ax, closefig)
    if return_artists:
        return fig, ax, im, assignments_text, theta_text
    else:
        return fig


def plot_quantity(
    quantity=None, max_quantity=None, ax=None, filename=None, label="", nseen=None
):
    if isinstance(quantity, pd.Series):
        quantity = quantity.values
    n_sqrt = int(np.sqrt(quantity.size))
    if n_sqrt ** 2 == quantity.size:
        quantity = quantity.reshape((n_sqrt, n_sqrt)).T
        figsize = (8, 7)
        oned = False
    else:
        quantity = quantity.reshape((1, quantity.size))
        figsize = (8, 3)
        oned = True
    fig, ax, closefig = openfig(ax, figsize=figsize)
    if max_quantity is None:
        max_quantity = quantity.max() * 1.1
    im = ax.imshow(
        quantity, interpolation="nearest", vmin=0, vmax=max_quantity, cmap=cm.plasma
    )
    ax.yaxis.set_ticks([])
    if nseen is not None:
        ax.set_title(f"examples seen: {nseen: 6d}", loc="right")
    if oned:
        ax.xaxis.set_ticks(np.arange(quantity.size))
        cbar_aspect = 5
    else:
        ax.xaxis.set_ticks([])
        cbar_aspect = 20
    cbar = add_colorbar(im, aspect=cbar_aspect)
    cbar.set_label(label)
    endfig(filename, fig, ax, closefig)
    return fig


def plot_accuracy(acchist, ax=None, filename=None):
    fig, ax, closefig = openfig(ax, figsize=(6, 4.5))
    i = acchist.index
    amid, alow, ahigh, fnull, amid_exc, alow_exc, ahigh_exc, = acchist.values.T
    ax.fill_between(i, alow, ahigh, facecolor="blue", alpha=0.5)
    ax.plot(i, amid, color="blue")
    if (amid != amid_exc).any():
        ax.fill_between(i, alow_exc, ahigh_exc, facecolor="red", alpha=0.25)
        ax.plot(i, amid_exc, color="red")
        ax.plot(i, fnull, color="green", ls="--")
    ax.set_xlabel("examples seen")
    ax.set_ylabel("accuracy (mean and 95% conf. int.)")
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0, ymax=100)
    endfig(filename, fig, ax, closefig)
    return fig


def plot_theta_summary(thetahist, ax=None, filename=None, label=""):
    fig, ax, closefig = openfig(ax, figsize=(6, 4.5))
    thetahist = thetahist.groupby("nseen")
    tlow = thetahist.quantile(0.025)
    tmid = thetahist.quantile(0.5)
    thigh = thetahist.quantile(0.975)
    ax.fill_between(tmid.index, tlow, thigh, facecolor="blue", alpha=0.5)
    ax.plot(tmid.index, tmid, color="blue")
    ax.set_xlabel("examples seen")
    ax.set_ylabel(f"theta {label} (median and 95% range)")
    ax.set_xlim(xmin=0)
    endfig(filename, fig, ax, closefig)
    return fig


def plot_rates_summary(ratehist, ax=None, filename=None, label=""):
    fig, ax, closefig = openfig(ax, figsize=(6, 4.5))
    ratehist = ratehist.groupby("nseen")
    tlow = ratehist.quantile(0.025)
    tmid = ratehist.quantile(0.5)
    thigh = ratehist.quantile(0.975)
    ax.fill_between(tmid.index, tlow, thigh, facecolor="blue", alpha=0.5)
    ax.plot(tmid.index, tmid, color="blue")
    ax.set_xlabel("examples seen")
    ax.set_ylabel(f"spike rate {label} (median and 95% range)")
    ax.set_xlim(xmin=0)
    endfig(filename, fig, ax, closefig)
    return fig


def openfig(ax, figsize=None):
    closefig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        closefig = True
    else:
        fig = None
    return fig, ax, closefig


def endfig(filename, fig, ax, closefig, nseen=None):
    if filename is not None:
        f = ax.get_figure()
        f.savefig(filename)
        if nseen is not None:
            f.savefig(rreplace(filename, ".", f"-n{nseen:06d}."))
    if closefig:
        plt.close(fig)


def rreplace(s, old, new, occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def spike_counts_from_cumulative(
    cumulative_spike_counts, n_data, n_tbin, n_neurons, start=0, end=None, atmost=None
):
    log.debug("Producing spike counts from cumulative counts")
    # log.debug(f"cumulative_spike_counts:\n{cumulative_spike_counts}")
    if isinstance(cumulative_spike_counts, pd.DataFrame):
        cumulative_spike_counts = cumulative_spike_counts.values
    counts = np.diff(cumulative_spike_counts, axis=0)
    log.debug("Counts for {} examples".format(n_tbin))
    s = sparse.coo_matrix(counts)
    spikecounts = pd.DataFrame(
        {"tbin": s.row, "i": s.col, "count": s.data}, dtype=np.int32
    )
    if start is not None and start < 0:
        start += n_tbin
    if end is None:
        end = n_tbin
    elif end < 0:
        end += n_tbin
    if atmost is not None:
        if end is None:
            end = min(end, start + atmost)
        else:
            start = max(start, end - atmost)
    log.debug("Starting at tbin {}".format(start))
    spikecounts = spikecounts[spikecounts["tbin"] >= start]
    log.debug("Ending before tbin {}".format(end))
    spikecounts = spikecounts[spikecounts["tbin"] < end]
    idx = pd.MultiIndex.from_product(
        [np.arange(start, end), np.arange(n_neurons)], names=["tbin", "i"]
    )
    spikecounts = spikecounts.set_index(["tbin", "i"])
    spikecounts = spikecounts.reindex(idx, fill_value=0)
    spikecounts = spikecounts.reset_index()
    spikecounts["example_idx"] = spikecounts["tbin"] % n_data
    spikecounts = spikecounts.set_index(["tbin", "i"])
    return spikecounts


def get_assignments(counts, labels):
    counts = counts.reset_index("i").set_index("example_idx")
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
    null = predictions["count"] == 0
    predictions.loc[null, "assignment"] = -1
    return predictions


def get_accuracy(predictions, nseen):
    match = predictions["assignment"] == predictions["label"]
    k = match.sum()
    n = len(predictions)
    if n == 0:
        return None
    mid = 100 * k / n
    lower, upper = 100 * binom_conf_interval(k, n, conf=0.95)
    nnull = (predictions["assignment"] == -1).sum()
    fnull = 100 * nnull / n
    n_exc = n - nnull
    mid_exc = 100 * k / n_exc
    lower_exc, upper_exc = 100 * binom_conf_interval(k, n_exc, conf=0.95)
    return pd.DataFrame(
        {
            "mid": mid,
            "lower": lower,
            "upper": upper,
            "fnull": fnull,
            "mid_exc": mid_exc,
            "lower_exc": lower_exc,
            "upper_exc": upper_exc,
        },
        index=[nseen],
        dtype=np.float32,
    )


def add_nseen_index(df, nseen):
    df = df.set_axis([nseen * np.ones_like(df.index), df.index], inplace=False)
    df = df.rename_axis(["nseen", "i"])
    return df


def get_labels(data):
    return pd.DataFrame({"label": data["y"]}).rename_axis("example_idx")


def get_windows(nseen, progress_assignments_window, progress_accuracy_window):
    log.debug("Requested assignments window: {}".format(progress_assignments_window))
    log.debug("Requested accuracy window: {}".format(progress_accuracy_window))
    progress_window = progress_assignments_window + progress_accuracy_window
    if progress_window > nseen:
        log.debug(
            "Fewer examples have been seen than required for the requested progress windows."
        )
        if progress_assignments_window > 0:
            log.debug(
                "Discarding first 20% of available examples to avoid initial contamination."
            )
            log.debug("Dividing remaining examples in proportion to requested windows.")
            assignments_window = int(
                0.8 * nseen * progress_assignments_window / progress_window
            )
            accuracy_window = int(
                0.8 * nseen * progress_accuracy_window / progress_window
            )
        else:
            # if requested accuracy window is zero, then implies test mode: use all seen
            accuracy_window = nseen
    else:
        assignments_window = progress_assignments_window
        accuracy_window = progress_accuracy_window
    log.info("Used assignments window: {}".format(assignments_window))
    log.info("Used accuracy window: {}".format(accuracy_window))
    return assignments_window, accuracy_window


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


# copied from keras.utils.to_categorical
def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """

    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


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


def get_metadata(store):
    return store.root._v_attrs


def record_arguments(frame, values):
    args, _, _, _ = getargvalues(frame)
    args.remove("store")
    argdict = {}
    for a in args:
        argdict[a] = values[a]
    for a, v in argdict.items():
        log.info(f"{a}: {v}")
    return argdict


def create_test_store(storefilename, originalstorefilename):
    with pd.HDFStore(
        originalstorefilename, mode="r", complib="blosc", complevel=9
    ) as originalstore:
        with pd.HDFStore(
            storefilename, mode="w", complib="blosc", complevel=9
        ) as store:
            for k in originalstore.root._v_attrs._v_attrnamesuser:
                store.root._v_attrs[k] = originalstore.root._v_attrs[k]
            nseen = originalstore["nseen"].max()
            for k in originalstore.keys():
                if k.startswith(("/connections", "/assignments", "/theta")):
                    data = originalstore.select(k, where="nseen == nseen")
                    data.index.set_levels([0], level="nseen", inplace=True)
                    store.put(k, data, format="table")


def float_or_none(x):
    if not (x is None or x.lower() == "none"):
        return float(x)
