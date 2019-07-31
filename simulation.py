#!/usr/bin/env python
"""
Original Python2/Brian1 version created by Peter U. Diehl
on 2014-12-15, GitHub updated 2015-08-07
https://github.com/peter-u-diehl/stdp-mnist

Brian2 version created by Xu Zhang
GitHub updated 2016-09-13
https://github.com/zxzhijia/Brian2STDPMNIST

This version created by Steven P. Bamford
https://github.com/bamford/Brian2STDPMNIST

@author: Steven P. Bamford
"""

# conda create -y -n brian2 python=3
# conda install -y -n brian2 -c conda-forge numpy scipy matplotlib brian2 pandas ipython

import logging

logging.captureWarnings(True)
log = logging.getLogger("spiking-mnist")
log.setLevel(logging.DEBUG)

import os.path
import numpy as np
import pandas as pd
import brian2 as b2
import pickle
import time
import datetime
from inspect import getargvalues, currentframe, getframeinfo
import json

from utilities import (
    get_matrix_from_file,
    connections_to_file,
    get_metadata,
    get_labeled_data,
    to_categorical,
    get_labels,
    get_windows,
    spike_counts_from_cumulative,
    get_assignments,
    add_nseen_index,
    get_predictions,
    get_accuracy,
    plot_theta_summary,
    plot_quantity,
    plot_rates_summary,
    theta_to_pandas,
    plot_accuracy,
    connections_to_pandas,
    plot_weights,
)

from neurons import DiehlAndCookExcitatoryNeuronGroup, DiehlAndCookInhibitoryNeuronGroup
from synapses import DiehlAndCookSynapses

from IPython import embed


# b2.set_device('cpp_standalone', build_on_run=False)


class config:
    # a global object to store configuration info
    pass


def load_connections(connName, random=True):
    if random:
        path = config.random_weight_path
    else:
        path = config.weight_path
    filename = os.path.join(path, "{}.npy".format(connName))
    return get_matrix_from_file(filename)


def save_connections(connections, iteration=None):
    for connName in config.save_conns:
        log.info("Saving connections {}".format(connName))
        conn = connections[connName]
        filename = os.path.join(config.weight_path, "{}".format(connName))
        if iteration is not None:
            filename += "-{:06d}".format(iteration)
        connections_to_file(conn, filename)


def load_theta(population_name):
    log.info("Loading theta for population {}".format(population_name))
    filename = os.path.join(config.weight_path, "theta_{}.npy".format(population_name))
    return np.load(filename) * b2.volt


def save_theta(population_names, neuron_groups, iteration=None):
    log.info("Saving theta")
    for pop_name in population_names:
        filename = os.path.join(config.weight_path, "theta_{}".format(pop_name))
        if iteration is not None:
            filename += "-{:06d}".format(iteration)
        np.save(filename, neuron_groups[pop_name + "e"].theta)


def get_initial_weights(n_input, n_e):
    matrices = {}
    npr = np.random.RandomState(9728364)
    # for neuron group A
    matrices["AeAi"] = np.eye(n_e["A"]) * 10.4
    matrices["AiAe"] = 17.0 * (1 - np.eye(n_e["A"]))
    matrices["XeAe"] = npr.uniform(0.003, 0.303, (n_input["X"], n_e["A"]))
    # XeAi connections not currently used but this is how they appear to be
    # generated from inspection of pre-made weights supplied with DC15 code
    new = np.zeros((n_input["X"], n_e["A"]))
    n_connect = int(0.1 * n_input["X"] * n_e["A"])
    connect = npr.choice(n_input["X"] * n_e["A"], n_connect, replace=False)
    new.flat[connect] = npr.uniform(0.0, 0.2, n_connect)
    matrices["XeAi"] = new
    # for neuron group O --- TODO: refine
    matrices["OeOi"] = np.eye(n_e["O"]) * 10.4
    matrices["OiOe"] = 17.0 * (1 - np.eye(n_e["O"]))
    matrices["YeOe"] = np.eye(n_e["O"]) * 10.4
    # between neuron groups A and O --- TODO: refine
    # matrices["AeOe"] = npr.uniform(0.003, 0.303, (n_e["A"], n_e["O"]))
    matrices["AeOe"] = np.zeros((n_e["A"], n_e["O"])) + 0.1
    return matrices


def main(**kwargs):
    if kwargs["runname"] is None:
        if kwargs["resume"]:
            print(f"Must provide runname to resume")
            exit(2)
        kwargs["runname"] = datetime.datetime.now().replace(microsecond=0).isoformat()
    outputpath = os.path.join(kwargs["output"], kwargs["runname"])
    try:
        os.makedirs(
            outputpath,
            exist_ok=(kwargs["clobber"] or kwargs["resume"] or kwargs["test_mode"]),
        )
    except (OSError, FileExistsError):
        print(f"Refusing to overwrite existing output files in {outputpath}")
        print(f"Use --clobber to force overwriting")
        exit(8)
    suffix = ""
    if kwargs["test_mode"]:
        mode = "r"
        suffix = "_test"
    elif kwargs["resume"]:
        mode = "a"
    else:
        mode = "w"
    logfilename = os.path.join(outputpath, f"output{suffix}.log")
    fh = logging.FileHandler(logfilename, mode)
    fh.setLevel(logging.DEBUG if kwargs["debug"] else logging.INFO)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    storefilename = os.path.join(outputpath, "store.h5")
    with pd.HDFStore(storefilename, mode=mode, complib="blosc", complevel=9) as store:
        kwargs["store"] = store
        simulation(**kwargs)


def simulation(
    test_mode=True,
    runname=None,
    num_epochs=None,
    progress_interval=None,
    progress_assignments_window=None,
    progress_accuracy_window=None,
    record_spikes=False,
    permute_data=False,
    size=400,
    resume=False,
    stdp_rule="original",
    custom_namespace=None,
    use_premade_weights=False,
    profile=False,
    store=None,
    **kwargs,
):
    metadata = get_metadata(store)
    if not resume:
        metadata.nseen = 0
        metadata.nprogress = 0

    if test_mode:
        random_weights = False
        use_premade_weights = True
        ee_STDP_on = False
        if num_epochs is None:
            num_epochs = 1
        if progress_interval is None:
            progress_interval = 0
    else:
        random_weights = not (resume or use_premade_weights)
        ee_STDP_on = True
        if num_epochs is None:
            num_epochs = 3
        if progress_interval is None:
            progress_interval = 1000

    if progress_interval > 0:
        if progress_assignments_window is None:
            progress_assignments_window = 10000
        if progress_accuracy_window is None:
            progress_accuracy_window = 10000

    log.info("Brian2STDPMNIST/simulation.py")
    log.info("Arguments =============")
    args, _, _, values = getargvalues(currentframe())
    args.remove("store")
    for a in args:
        log.info(f"{a}: {locals()[a]}")
        metadata["args"] = args
    log.info("=======================")

    # load MNIST
    training, testing = get_labeled_data()
    config.classes = np.unique(training["y"])
    config.num_classes = len(config.classes)

    # configuration
    np.random.seed(0)
    modulefilename = getframeinfo(currentframe()).filename
    config.data_path = os.path.dirname(os.path.abspath(modulefilename))
    config.random_weight_path = os.path.join(config.data_path, "random/")
    runpath = os.path.join("runs", runname)
    config.weight_path = os.path.join(runpath, "weights/")
    os.makedirs(config.weight_path, exist_ok=True)
    if test_mode:
        log.info("Testing run {}".format(runname))
    elif resume:
        log.info("Resuming training run {}".format(runname))
    else:
        log.info("Training run {}".format(runname))

    config.output_path = os.path.join(runpath, "output/")
    os.makedirs(config.output_path, exist_ok=True)

    if test_mode:
        data = testing
    else:
        data = training

    if permute_data:
        sample = np.random.permutation(len(data["y"]))
        data["x"] = data["x"][sample]
        data["y"] = data["y"][sample]

    num_examples = int(len(data["y"]) * num_epochs)
    n_input = data["x"][0].size
    n_data = data["y"].size
    if num_epochs < 1:
        n_data = int(np.ceil(n_data * num_epochs))
        data["x"] = data["x"][:n_data]
        data["y"] = data["y"][:n_data]

    # -------------------------------------------------------------------------
    # set parameters and equations
    # -------------------------------------------------------------------------
    # log.info('Original defaultclock.dt = {}'.format(str(b2.defaultclock.dt)))
    b2.defaultclock.dt = 0.5 * b2.ms
    log.info("defaultclock.dt = {}".format(str(b2.defaultclock.dt)))

    n_e = {"A": size, "O": config.num_classes}
    n_i = n_e

    single_example_time = 0.35 * b2.second
    resting_time = 0.15 * b2.second
    total_example_time = single_example_time + resting_time
    runtime = num_examples * total_example_time

    input_population_names = ["X", "Y"]
    population_names = ["A", "O"]
    connection_names = ["XA", "YO", "AO"]
    config.save_conns = ["XeAe", "YeOe", "AeOe"]
    config.plot_conns = ["XeAe", "AeOe"]
    forward_conntype_names = ["ee"]
    recurrent_conntype_names = ["ei_rec", "ie_rec"]
    stdp_conn_names = ["XeAe", "AeOe"]

    total_weight = {}
    total_weight["XeAe"] = 78.0
    total_weight["AeOe"] = n_e["A"] / 10.0  # TODO: refine?

    delay = {}  # TODO: potentially specify by connName?
    delay["ee"] = (0 * b2.ms, 10 * b2.ms)
    delay["ei"] = (0 * b2.ms, 5 * b2.ms)
    delay["ei_rec"] = (0 * b2.ms, 0 * b2.ms)
    delay["ie_rec"] = (0 * b2.ms, 0 * b2.ms)

    input_intensity = 2.0
    if test_mode:
        input_label_intensity = 0.0
    else:
        input_label_intensity = 10.0

    n_input = {"X": n_input, "Y": config.num_classes}

    initial_weight_matrices = get_initial_weights(n_input, n_e)

    theta_init = {"O": 0.0 * b2.mV}

    neuron_groups = {}
    connections = {}
    spike_monitors = {}
    state_monitors = {}
    network_operations = []

    # -------------------------------------------------------------------------
    # create network population and recurrent connections
    # -------------------------------------------------------------------------
    for subgroup_n, name in enumerate(population_names):
        log.info(f"Creating neuron group {name}")
        subpop_e = name + "e"
        subpop_i = name + "i"
        nge = neuron_groups[subpop_e] = DiehlAndCookExcitatoryNeuronGroup(
            n_e[name], test_mode=test_mode
        )
        ngi = neuron_groups[subpop_i] = DiehlAndCookInhibitoryNeuronGroup(n_i[name])

        if not random_weights:
            neuron_groups[subpop_e].theta = load_theta(name)
        elif name in theta_init:
            neuron_groups[subpop_e].theta = theta_init[name]

        for connType in recurrent_conntype_names:
            log.info(f"Creating recurrent connections for {connType}")
            preName = name + connType[0]
            postName = name + connType[1]
            connName = preName + postName
            conn = connections[connName] = DiehlAndCookSynapses(
                neuron_groups[preName], neuron_groups[postName], conn_type=connType
            )
            conn.connect()  # all-to-all connection
            minDelay, maxDelay = delay[connType]
            if maxDelay > 0:
                deltaDelay = maxDelay - minDelay
                conn.delay = "minDelay + rand() * deltaDelay"
            # TODO: the use of connections with fixed zero weights is inefficient
            # "random" connections for AeAi is matrix with zero everywhere
            # except the diagonal, which contains 10.4
            # "random" connections for AiAe is matrix with 17.0 everywhere
            # except the diagonal, which contains zero
            # TODO: these weights appear to have been tuned,
            #       we may need different values for the O layer
            if use_premade_weights:
                weightMatrix = load_connections(connName, random=True)
            else:
                log.info("Using generated initial weight matrices")
                weightMatrix = initial_weight_matrices[connName]
            conn.w = weightMatrix.flatten()

        log.debug(f"Creating spike monitors for {name}")
        spike_monitors[subpop_e] = b2.SpikeMonitor(nge, record=record_spikes)
        spike_monitors[subpop_i] = b2.SpikeMonitor(ngi, record=record_spikes)

    # -------------------------------------------------------------------------
    # create TimedArray of rates for input examples
    # -------------------------------------------------------------------------
    input_dt = 50 * b2.ms
    n_dt_example = int(round(single_example_time / input_dt))
    n_dt_rest = int(round(resting_time / input_dt))
    n_dt_total = int(n_dt_example + n_dt_rest)
    input_rates = np.zeros((n_data * n_dt_total, n_input["X"]), dtype=np.float16)
    log.info("Preparing input rate stream {}".format(input_rates.shape))
    for j in range(n_data):
        spike_rates = data["x"][j].reshape(n_input["X"]) / 8
        spike_rates *= input_intensity
        start = j * n_dt_total
        input_rates[start : start + n_dt_example] = spike_rates
    input_rates = input_rates * b2.Hz
    stimulus_X = b2.TimedArray(input_rates, dt=input_dt)
    total_data_time = n_data * n_dt_total * input_dt

    # -------------------------------------------------------------------------
    # create TimedArray of rates for input labels
    # -------------------------------------------------------------------------
    input_label_rates = np.zeros((n_data * n_dt_total, n_input["Y"]), dtype=np.float16)
    log.info("Preparing input label rate stream {}".format(input_label_rates.shape))
    label_spike_rates = to_categorical(data["y"], dtype=np.float16)
    label_spike_rates *= input_label_intensity
    for j in range(n_data):
        start = j * n_dt_total
        input_label_rates[start : start + n_dt_example] = label_spike_rates[j]
    input_label_rates = input_label_rates * b2.Hz
    stimulus_Y = b2.TimedArray(input_label_rates, dt=input_dt)

    # -------------------------------------------------------------------------
    # create input population and connections from input populations
    # -------------------------------------------------------------------------
    for k, name in enumerate(input_population_names):
        subpop_e = name + "e"
        # stimulus is repeated for duration of simulation
        # (i.e. if there are multiple epochs)
        neuron_groups[subpop_e] = b2.PoissonGroup(
            n_input[name], rates=f"stimulus_{name}(t % total_data_time, i)"
        )
        log.debug(f"Creating spike monitors for {name}")
        spike_monitors[subpop_e] = b2.SpikeMonitor(
            neuron_groups[subpop_e], record=record_spikes
        )

    for name in connection_names:
        log.info(f"Creating connections between {name[0]} and {name[1]}")
        for connType in forward_conntype_names:
            log.debug(f"connType {connType}")
            preName = name[0] + connType[0]
            postName = name[1] + connType[1]
            connName = preName + postName
            stdp_on = ee_STDP_on and connName in stdp_conn_names
            conn = connections[connName] = DiehlAndCookSynapses(
                neuron_groups[preName],
                neuron_groups[postName],
                conn_type=connType,
                stdp_on=stdp_on,
                stdp_rule=stdp_rule,
                custom_namespace=custom_namespace,
            )
            conn.connect()  # all-to-all connection
            minDelay, maxDelay = delay[connType]
            if maxDelay > 0:
                deltaDelay = maxDelay - minDelay
                conn.delay = "minDelay + rand() * deltaDelay"
            if use_premade_weights:
                weightMatrix = load_connections(connName, random=random_weights)
            else:
                log.info("Using generated initial weight matrices")
                weightMatrix = initial_weight_matrices[connName]
            conn.w = weightMatrix.flatten()

    if ee_STDP_on:

        @b2.network_operation(dt=total_example_time)
        def normalize_weights(t):
            for connName in connections:
                if connName in stdp_conn_names:
                    # log.debug(
                    #     "Normalizing weights for {} " "at time {}".format(connName, t)
                    # )
                    conn = connections[connName]
                    connweights = np.reshape(
                        conn.w, (len(conn.source), len(conn.target))
                    )
                    colSums = connweights.sum(axis=0)
                    ok = colSums > 0
                    colFactors = np.ones_like(colSums)
                    colFactors[ok] = total_weight[connName] / colSums[ok]
                    connweights *= colFactors
                    conn.w = connweights.flatten()

        network_operations.append(normalize_weights)

    @b2.network_operation(dt=total_example_time)
    def record_cumulative_spike_counts(t):
        for name in population_names + input_population_names:
            subpop_e = name + "e"
            count = pd.DataFrame(
                spike_monitors[subpop_e].count[:][None, :], index=[metadata.nseen]
            )
            count = count.rename_axis("tbin")
            count = count.rename_axis("neuron", axis="columns")
            store.append(f"cumulative_spike_counts/{subpop_e}", count)
        if t > 0:
            metadata.nseen += 1

    network_operations.append(record_cumulative_spike_counts)

    if progress_interval > 0:

        @b2.network_operation(dt=total_example_time * progress_interval)
        def progress(t):
            # if t < total_example_time:
            #    return None
            log.debug("Starting progress")
            starttime = time.process_time()
            labels = get_labels(data)
            log.info("So far seen {} examples".format(metadata.nseen))
            store.append(
                f"nseen", pd.Series(data=[metadata.nseen], index=[metadata.nprogress])
            )
            metadata.nprogress += 1
            assignments_window, accuracy_window = get_windows(
                metadata.nseen, progress_assignments_window, progress_accuracy_window
            )
            for name in population_names + input_population_names:
                log.info(f"Progress for population {name}")
                subpop_e = name + "e"
                csc = store.select(f"cumulative_spike_counts/{subpop_e}")
                spikecounts_present = spike_counts_from_cumulative(
                    csc, n_data, start=-accuracy_window
                )
                n_spikes_present = len(spikecounts_present)
                if n_spikes_present > 0:
                    spikerates = (
                        spikecounts_present.groupby("i")["count"]
                        .mean()
                        .astype(np.float32)
                    )
                    spikerates = spikerates.reindex(
                        np.arange(len(spikerates)), fill_value=0
                    )
                    spikerates = add_nseen_index(spikerates, metadata.nseen)
                    store.append(f"rates/{subpop_e}", spikerates)
                    store.flush()
                    fn = os.path.join(
                        config.output_path, "spikerates-summary-{}.pdf".format(subpop_e)
                    )
                    plot_rates_summary(store.select(f"rates/{subpop_e}"), filename=fn)
                if name in population_names:
                    spikecounts_past = spike_counts_from_cumulative(
                        csc, n_data, end=-accuracy_window, atmost=assignments_window
                    )
                    log.debug(
                        "Assignments based on {} spikes".format(len(spikecounts_past))
                    )
                    if name == "O":
                        assignments = pd.DataFrame(
                            {"label": np.arange(n_e[name], dtype=np.int32)}
                        )
                    else:
                        assignments = get_assignments(spikecounts_past, labels)
                    assignments = add_nseen_index(assignments, metadata.nseen)
                    store.append(f"assignments/{subpop_e}", assignments)
                    if n_spikes_present == 0:
                        log.debug(
                            "No spikes in present interval - skipping accuracy estimate"
                        )
                    else:
                        log.debug(
                            "Accuracy based on {} spikes".format(n_spikes_present)
                        )
                        predictions = get_predictions(
                            spikecounts_present, assignments, labels
                        )
                        accuracy = get_accuracy(predictions, metadata.nseen)
                        store.append(f"accuracy/{subpop_e}", accuracy)
                        store.flush()
                        log.info(
                            "Accuracy [{}]: {:.1f}%  ({:.1f}–{:.1f}% 1σ conf. int.)".format(
                                subpop_e, *accuracy.values.flat
                            )
                        )
                        fn = os.path.join(
                            config.output_path, "accuracy-{}.pdf".format(subpop_e)
                        )
                        plot_accuracy(store.select(f"accuracy/{subpop_e}"), filename=fn)
                        fn = os.path.join(
                            config.output_path, "spikerates-{}.pdf".format(subpop_e)
                        )
                        plot_quantity(
                            spikerates,
                            filename=fn,
                            label=f"spike rate {subpop_e}",
                            nseen=metadata.nseen,
                        )
                    theta = theta_to_pandas(subpop_e, neuron_groups, metadata.nseen)
                    store.append(f"theta/{subpop_e}", theta)
                    fn = os.path.join(
                        config.output_path, "theta-{}.pdf".format(subpop_e)
                    )
                    plot_quantity(
                        theta,
                        filename=fn,
                        label=f"theta {subpop_e} (mV)",
                        nseen=metadata.nseen,
                    )
                    fn = os.path.join(
                        config.output_path, "theta-summary-{}.pdf".format(subpop_e)
                    )
                    plot_theta_summary(store.select(f"theta/{subpop_e}"), filename=fn)
            for conn in config.save_conns:
                log.info(f"Progress for connection {conn}")
                conn_df = connections_to_pandas(connections[conn], metadata.nseen)
                store.append(f"connections/{conn}", conn_df)
            for conn in config.plot_conns:
                subpop = conn[-2:]
                if "O" in subpop:
                    assignments = None
                else:
                    try:
                        assignments = store.select(
                            f"assignments/{subpop}", where="nseen == metadata.nseen"
                        )
                        assignments = assignments.reset_index("nseen", drop=True)
                    except KeyError:
                        assignments = None
                fn = os.path.join(config.output_path, "weights-{}.pdf".format(conn))
                plot_weights(
                    connections[conn],
                    assignments,
                    theta=None,
                    filename=fn,
                    max_weight=None,
                    nseen=metadata.nseen,
                    output=("O" in subpop),
                    label=subpop,
                )

            log.debug(
                "progress took {:.3f} seconds".format(time.process_time() - starttime)
            )

        network_operations.append(progress)

    # -------------------------------------------------------------------------
    # run the simulation and set inputs
    # -------------------------------------------------------------------------
    log.info("Constructing the network")
    net = b2.Network()
    for obj_list in [neuron_groups, connections, spike_monitors, state_monitors]:
        for key in obj_list:
            net.add(obj_list[key])

    for obj in network_operations:
        net.add(obj)

    log.info("Starting simulations")

    net.run(runtime, report="text", report_period=(60 * b2.second), profile=profile)

    b2.device.build(
        directory=os.path.join("build", runname), compile=True, run=True, debug=False
    )

    if profile:
        log.debug(b2.profiling_summary(net, 10))

    # -------------------------------------------------------------------------
    # save results
    # -------------------------------------------------------------------------

    log.info("Saving results")
    if not test_mode:
        save_theta(population_names, neuron_groups)
        save_connections(connections)

    saveobj = {
        "spike_monitors": {km: vm.get_states() for km, vm in spike_monitors.items()},
        "state_monitors": {km: vm.get_states() for km, vm in state_monitors.items()},
        "labels": data["y"],
        "total_example_time": total_example_time,
        "dt": b2.defaultclock.dt,
    }
    with open(os.path.join(config.output_path, "saved.pickle"), "wb") as f:
        pickle.dump(saveobj, f)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Brian2 implementation of Diehl & Cook 2015, "
            "an MNIST classifer constructed from a "
            "Spiking Neural Network with STDP-based learning."
        )
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--test", dest="test_mode", action="store_true", help="Enable test mode"
    )
    mode_group.add_argument(
        "--train", dest="test_mode", action="store_false", help="Enable train mode"
    )
    parser.add_argument(
        "--runname",
        type=str,
        default=None,
        help="Name of output folder, if none given defaults to date and time.",
    )
    parser.add_argument(
        "--output", type=str, default="./runs/", help="Parent path for output folder"
    )
    debug_group = parser.add_mutually_exclusive_group(required=False)
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=argparse.SUPPRESS,  # default to debug=True
        help="Include debug output from log file",
    )
    debug_group.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Omit debug output in log file",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Force overwrite of files in existing run folder",
    )
    parser.add_argument("--num_epochs", type=float, default=None)
    parser.add_argument("--progress_interval", type=int, default=None)
    parser.add_argument("--assignments_window", type=int, default=None)
    parser.add_argument("--accuracy_window", type=int, default=None)
    parser.add_argument("--record_spikes", action="store_true")
    parser.add_argument("--permute_data", action="store_true")
    parser.add_argument(
        "--size",
        type=int,
        default=400,
        help="""Number of neurons in the computational layer.
                Currently this must be a square number.""",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue on from existing run."
    )

    parser.add_argument(
        "--stdp_rule",
        type=str,
        default="original",
        choices=[
            "original",
            "minimal-triplet",
            "full-triplet",
            "powerlaw",
            "exponential",
            "symmetric",
        ],
    )
    parser.add_argument(
        "--custom_namespace",
        type=str,
        default="{}",
        help=(
            "Customise the synapse namespace. "
            "This should be given as a dictionary, surrounded by quotes, "
            'for example: \'{"tar": 0.1, "mu": 2.0}\'.'
        ),
    )
    parser.add_argument("--use_premade_weights", action="store_true")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()

    custom_namespace_arg = json.loads(args.custom_namespace.replace("'", '"'))

    sys.exit(
        main(
            test_mode=args.test_mode,
            runname=args.runname,
            output=args.output,
            debug=args.debug,
            clobber=args.clobber,
            num_epochs=args.num_epochs,
            progress_interval=args.progress_interval,
            progress_assignments_window=args.assignments_window,
            progress_accuracy_window=args.accuracy_window,
            record_spikes=args.record_spikes,
            permute_data=args.permute_data,
            size=args.size,
            resume=args.resume,
            stdp_rule=args.stdp_rule,
            custom_namespace=custom_namespace_arg,
            use_premade_weights=args.use_premade_weights,
            profile=args.profile,
        )
    )
