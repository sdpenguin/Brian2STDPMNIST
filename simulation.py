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

# conda install -c conda-forge numpy scipy matplotlib keras brian2
# conda install -c brian-team brian2tools

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

from utilities import *

from neurons import DiehlAndCookExcitatoryNeuronGroup, DiehlAndCookInhibitoryNeuronGroup

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
    runname="",
    num_epochs=None,
    record_spikes=False,
    progress_interval=None,
    save_interval=None,
    profile=False,
):

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
            num_epochs = 1
        if save_interval is None:
            save_interval = 1000
        if progress_interval is None:
            progress_interval = 100

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

    n_e = 400
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

    namespace = {
        "tc_pre_ee": 20 * b2.ms,
        "tc_post_1_ee": 20 * b2.ms,
        "tc_post_2_ee": 40 * b2.ms,
        "nu_ee_pre": 0.0001,
        "nu_ee_post": 0.01,
        "wmax_ee": 1.0,
        "exp_ee_pre": 0.2,
        "exp_ee_post": 0.2,
    }

    eqs_stdp_ee = """
            post2before  : 1
            dpre/dt = -pre/(tc_pre_ee)  : 1 (event-driven)
            dpost1/dt  = -post1/(tc_post_1_ee)  : 1 (event-driven)
            dpost2/dt  = -post2/(tc_post_2_ee)  : 1 (event-driven)
            """
    eqs_stdp_pre_ee = "pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)"
    eqs_stdp_post_ee = "post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1."

    input_intensity = 2.0

    n_pop = len(population_names)

    neuron_groups = {}
    input_groups = {}
    connections = {}
    spike_monitors = {}
    state_monitors = {}
    network_operations = []
    cumulative_spike_counts = {}

    neuron_groups["e"] = DiehlAndCookExcitatoryNeuronGroup(n_e * n_pop,
                                                           test_mode=test_mode)
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
            # "random" connections for AeAi is matrix with zero everywhere
            # except the diagonal, which contains 10.4
            # "random" connections for AiAe is matrix with 17.0 everywhere
            # except the diagonal, which contains zero
            weightMatrix = load_connections(connName, random=True)
            model = "w : 1"
            pre = "g%s_post += w" % connType[0]
            post = ""
            if ee_STDP_on:
                if "ee" in recurrent_conntype_names:
                    log.info(f"Creating STDP for connection {name[0]}e{name[1]}e")
                    model += eqs_stdp_ee
                    pre += "; " + eqs_stdp_pre_ee
                    post = eqs_stdp_post_ee
            conn = connections[connName] = b2.Synapses(
                neuron_groups[preName],
                neuron_groups[postName],
                model=model,
                on_pre=pre,
                on_post=post,
                namespace=namespace
            )
            conn.connect()  # all-to-all connection
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
        input_groups[subpop_e] = b2.PoissonGroup(
            n_input, rates="stimulus(t%total_data_time, i)"
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
            weightMatrix = load_connections(connName, random=random_weights)
            model = "w : 1"
            pre = "g%s_post += w" % connType[0]
            post = ""
            if ee_STDP_on:
                log.info(f"Creating STDP for connection {name[0]}e{name[1]}e")
                model += eqs_stdp_ee
                pre += "; " + eqs_stdp_pre_ee
                post = eqs_stdp_post_ee

            conn = connections[connName] = b2.Synapses(
                input_groups[preName],
                neuron_groups[postName],
                model=model,
                on_pre=pre,
                on_post=post,
                namespace=namespace
            )
            conn.connect()  # all-to-all connection
            minDelay = delay[connType][0]
            maxDelay = delay[connType][1]
            deltaDelay = maxDelay - minDelay
            conn.delay = "minDelay + rand() * deltaDelay"
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
                        csc, end=-progress_interval, atmost=100 * progress_interval
                    )
                    log.debug(
                        "Assignments based on {} spikes".format(len(spikecounts_past))
                    )
                    assignments = get_assignments(spikecounts_past, labels)
                    spikecounts_present = spike_counts_from_cumulative(
                        csc, start=-progress_interval
                    )
                    log.debug(
                        "Accuracy based on {} spikes".format(len(spikecounts_present))
                    )
                    predictions = get_predictions(
                        spikecounts_present, assignments, labels
                    )
                    accuracy = get_accuracy(predictions)
                    accuracy *= 100
                    progress_accuracy[subpop_e][nseen] = accuracy
                    print(
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
        print(b2.profiling_summary(net, 10))

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
    parser.add_argument("--runname", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--num_epochs", default=None)
    parser.add_argument("--progress_interval", default=None)
    parser.add_argument("--save_interval", default=None)
    parser.add_argument("--record_spikes", action="store_true")

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
        )
    )
