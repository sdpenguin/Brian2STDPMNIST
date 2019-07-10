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
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)

import os.path
import numpy as np
import brian2 as b2
import pickle
import time
import datetime
from inspect import getargvalues, currentframe

from utilities import *

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
    filename = os.path.join(config.data_path, path, "{}.npy".format(connName))
    return get_matrix_from_file(filename)


def save_connections(connections, iteration=None):
    for connName in config.save_conns:
        log.info("Saving connections {}".format(connName))
        conn = connections[connName]
        filename = os.path.join(
            config.data_path, config.weight_path, "{}".format(connName)
        )
        if iteration is not None:
            filename += "-{:06d}".format(iteration)
        connections_to_file(conn, filename)


def load_theta(population_name):
    log.info("Loading theta for population {}".format(population_name))
    filename = os.path.join(
        config.data_path, config.weight_path, "theta_{}.npy".format(population_name)
    )
    return np.load(filename) * b2.volt


def save_theta(population_names, neuron_groups, iteration=None):
    log.info("Saving theta")
    for pop_name in population_names:
        filename = os.path.join(
            config.data_path, config.weight_path, "theta_{}".format(pop_name)
        )
        if iteration is not None:
            filename += "-{:06d}".format(iteration)
        np.save(filename, neuron_groups[pop_name + "e"].theta)


def main(
    test_mode=True,
    runname=None,
    num_epochs=None,
    record_spikes=False,
    progress_interval=None,
    save_interval=None,
    profile=False,
    permute_data=False,
    stdp_rule="original",
    size=400,
):

    if runname is None:
        runname = datetime.datetime.now().replace(microsecond=0).isoformat()

    log.info('Brian2STDPMNIST/simulation.py')
    log.info('Arguments =============')
    args, _, _, values = getargvalues(currentframe())
    for a in args:
        log.info(f'{a}: {values[a]}')
    log.info('=======================')

    # load MNIST
    training, testing = get_labeled_data()
    config.classes = np.unique(training["y"])
    config.num_classes = len(config.classes)

    # configuration
    np.random.seed(0)
    config.data_path = "./"
    config.random_weight_path = "random/"
    config.weight_path = os.path.join("runs", runname, "weights/")
    os.makedirs(config.weight_path, exist_ok=True)
    log.info("Running {}".format(runname))
    config.output_path = os.path.join("runs", runname, "output/")
    os.makedirs(config.output_path, exist_ok=True)

    if test_mode:
        data = testing
        random_weights = False
        ee_STDP_on = False
        if num_epochs is None:
            num_epochs = 1
        if save_interval is None:
            save_interval = 0
        if progress_interval is None:
            progress_interval = 0
    else:
        data = training
        random_weights = True
        ee_STDP_on = True
        if num_epochs is None:
            num_epochs = 3
        if save_interval is None:
            save_interval = 10000
        if progress_interval is None:
            progress_interval = 1000

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

    n_e = size
    n_i = n_e

    single_example_time = 0.35 * b2.second
    resting_time = 0.15 * b2.second
    total_example_time = single_example_time + resting_time
    runtime = num_examples * total_example_time

    input_population_names = ["X"]
    population_names = ["A"]
    input_connection_names = ["XA"]
    config.save_conns = ["XeAe"]
    input_conntype_names = ["ee_input"]
    recurrent_conntype_names = ["ei", "ie"]

    total_weight = {}
    total_weight["ee_input"] = 78.0

    delay = {}
    delay["ee_input"] = (0 * b2.ms, 10 * b2.ms)
    delay["ei_input"] = (0 * b2.ms, 5 * b2.ms)

    input_intensity = 2.0

    initial_weight_matrices = get_initial_weights(n_input, n_e)
    use_premade_weights = (n_e == 400)

    n_pop = len(population_names)

    neuron_groups = {}
    input_groups = {}
    connections = {}
    spike_monitors = {}
    state_monitors = {}
    network_operations = []
    cumulative_spike_counts = {}

    neuron_groups["e"] = DiehlAndCookExcitatoryNeuronGroup(
        n_e * n_pop, test_mode=test_mode
    )
    neuron_groups["i"] = DiehlAndCookInhibitoryNeuronGroup(n_i * n_pop)

    # -------------------------------------------------------------------------
    # create network population and recurrent connections
    # -------------------------------------------------------------------------
    for subgroup_n, name in enumerate(population_names):
        log.info(f"Creating neuron group {name}")
        subpop_e = name + "e"
        subpop_i = name + "i"
        nge = neuron_groups[subpop_e] = neuron_groups["e"][
            subgroup_n * n_e : (subgroup_n + 1) * n_e
        ]
        ngi = neuron_groups[subpop_i] = neuron_groups["i"][
            subgroup_n * n_i : (subgroup_n + 1) * n_i
        ]

        if not random_weights:
            neuron_groups["e"].theta = load_theta(name)

        for connType in recurrent_conntype_names:
            log.info(f"Creating recurrent connections for {connType}")
            preName = name + connType[0]
            postName = name + connType[1]
            connName = preName + postName
            conn = connections[connName] = DiehlAndCookSynapses(
                neuron_groups[preName], neuron_groups[postName], conn_type=connType
            )
            conn.connect()  # all-to-all connection
            # "random" connections for AeAi is matrix with zero everywhere
            # except the diagonal, which contains 10.4
            # "random" connections for AiAe is matrix with 17.0 everywhere
            # except the diagonal, which contains zero
            if use_premade_weights:
                weightMatrix = load_connections(connName, random=True)
            else:
                log.info('Using generated initial weight matrices')
                weightMatrix = initial_weight_matrices[connName]
            conn.w = weightMatrix.flatten()

        log.debug(f"Creating spike monitors for {name}")
        spike_monitors[subpop_e] = b2.SpikeMonitor(nge, record=record_spikes)
        spike_monitors[subpop_i] = b2.SpikeMonitor(ngi, record=record_spikes)

        cumulative_spike_counts[subpop_e] = []

    # -------------------------------------------------------------------------
    # create TimedArray of rates for input examples
    # -------------------------------------------------------------------------
    input_dt = 50 * b2.ms
    n_dt_example = int(round(single_example_time / input_dt))
    n_dt_rest = int(round(resting_time / input_dt))
    n_dt_total = int(n_dt_example + n_dt_rest)
    input_rates = np.zeros((n_data * n_dt_total, n_input), dtype=np.float16)
    log.info("Preparing input rate stream {}".format(input_rates.shape))
    for j in range(n_data):
        spike_rates = data["x"][j].reshape(n_input) / 8
        spike_rates *= input_intensity
        start = j * n_dt_total
        input_rates[start : start + n_dt_example] = spike_rates
    input_rates = input_rates * b2.Hz
    stimulus = b2.TimedArray(input_rates, dt=input_dt)
    total_data_time = n_data * n_dt_total * input_dt

    # -------------------------------------------------------------------------
    # create input population and connections from input populations
    # -------------------------------------------------------------------------
    for k, name in enumerate(input_population_names):
        subpop_e = name + "e"
        # stimulus is repeated for duration of simulation
        # (i.e. if there are multiple epochs)
        input_groups[subpop_e] = b2.PoissonGroup(
            n_input, rates="stimulus(t % total_data_time, i)"
        )
        log.debug(f"Creating spike monitors for {name}")
        spike_monitors[subpop_e] = b2.SpikeMonitor(
            input_groups[subpop_e], record=record_spikes
        )

    for name in input_connection_names:
        log.info(f"Creating connections between {name[0]} and {name[1]}")
        for connType in input_conntype_names:
            log.debug(f"connType {connType}")
            preName = name[0] + connType[0]
            postName = name[1] + connType[1]
            connName = preName + postName
            conn = connections[connName] = DiehlAndCookSynapses(
                input_groups[preName],
                neuron_groups[postName],
                conn_type=connType,
                stdp_on=ee_STDP_on,
                stdp_rule=stdp_rule,
            )
            conn.connect()  # all-to-all connection
            minDelay = delay[connType][0]
            maxDelay = delay[connType][1]
            deltaDelay = maxDelay - minDelay
            conn.delay = "minDelay + rand() * deltaDelay"
            if use_premade_weights:
                weightMatrix = load_connections(connName, random=random_weights)
            else:
                log.info('Using generated initial weight matrices')
                weightMatrix = initial_weight_matrices[connName]
            conn.w = weightMatrix.flatten()

    if ee_STDP_on:

        @b2.network_operation(dt=total_example_time)
        def normalize_weights(t):
            for connName in connections:
                if connName[1] == "e" and connName[3] == "e":
                    # log.debug('Normalizing weights for {} '
                    #          'at time {}'.format(connName, t))
                    conn = connections[connName]
                    connweights = np.reshape(
                        conn.w, (len(conn.source), len(conn.target))
                    )
                    colSums = connweights.sum(axis=0)
                    colFactors = total_weight["ee_input"] / colSums
                    connweights *= colFactors
                    conn.w = connweights.flatten()

        network_operations.append(normalize_weights)

    @b2.network_operation(dt=total_example_time)
    def record_cumulative_spike_counts(t):
        for name in population_names:
            subpop_e = name + "e"
            count = spike_monitors[subpop_e].count[:]
            cumulative_spike_counts[subpop_e].append(count)

    network_operations.append(record_cumulative_spike_counts)

    if save_interval > 0:

        @b2.network_operation(dt=total_example_time * save_interval)
        def save_status(t):
            if t > 0:
                log.debug("Starting save_status")
                start = time.process_time()
                tbin = int(t / total_example_time)
                save_theta(population_names, neuron_groups, tbin)
                save_connections(connections, tbin)
                log.debug(
                    "save_status took {:.3f} seconds".format(
                        time.process_time() - start
                    )
                )

        network_operations.append(save_status)

    if progress_interval > 0:
        progress_accuracy = {name + "e": {} for name in population_names}

        @b2.network_operation(dt=total_example_time * progress_interval)
        def progress(t):
            if t > (total_example_time * progress_interval * 1.5):
                log.debug("Starting progress")
                start = time.process_time()
                labels = get_labels(data)
                for name in population_names:
                    subpop_e = name + "e"
                    csc = cumulative_spike_counts[subpop_e]
                    nseen = len(csc) - 1
                    log.debug("So far seen {} examples".format(nseen))
                    spikecounts_past = spike_counts_from_cumulative(
                        csc,
                        n_data,
                        end=-progress_interval,
                        atmost=100 * progress_interval,
                    )
                    log.debug(
                        "Assignments based on {} spikes".format(len(spikecounts_past))
                    )
                    assignments = get_assignments(spikecounts_past, labels)
                    spikecounts_present = spike_counts_from_cumulative(
                        csc, n_data, start=-progress_interval
                    )
                    n_spikes_present = len(spikecounts_present)
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
                        accuracy = get_accuracy(predictions)
                        progress_accuracy[subpop_e][nseen] = accuracy
                        log.info(
                            "Accuracy [{}]: {:.1f}%  ({:.1f}–{:.1f}% 1σ conf. int.)".format(
                                subpop_e, *accuracy
                            )
                        )
                        fn = os.path.join(
                            config.output_path, "accuracy-{}.pdf".format(subpop_e)
                        )
                        plot_accuracy(progress_accuracy[subpop_e], filename=fn)

                fn = os.path.join(config.output_path, "weights.pdf")
                plot_weights(
                    connections["XeAe"], assignments, filename=fn, max_weight=None
                )

                log.debug(
                    "progress took {:.3f} seconds".format(time.process_time() - start)
                )

        network_operations.append(progress)

    # -------------------------------------------------------------------------
    # run the simulation and set inputs
    # -------------------------------------------------------------------------
    log.info("Constructing the network")
    net = b2.Network()
    primary_neuron_groups = {p: neuron_groups[p] for p in neuron_groups if len(p) == 1}
    for obj_list in [
        primary_neuron_groups,
        input_groups,
        connections,
        spike_monitors,
        state_monitors,
    ]:
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
    parser.add_argument("--runname", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--record_spikes", action="store_true")
    parser.add_argument("--permute_data", action="store_true")
    parser.add_argument(
        "--stdp_rule",
        type = str,
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
    parser.add_argument("--size", type=int, default=400)

    args = parser.parse_args()

    sys.exit(
        main(
            test_mode=args.test_mode,
            runname=args.runname,
            num_epochs=args.num_epochs,
            record_spikes=args.record_spikes,
            progress_interval=args.progress_interval,
            save_interval=args.save_interval,
            profile=args.profile,
            permute_data=args.permute_data,
            stdp_rule=args.stdp_rule,
            size=args.size,
        )
    )
