#!/usr/bin/env python
"""
Original Python2/Brian1 version created by Peter U. Diehl
on 2014-12-15, GitHub updated 2015-08-07
https://github.com/peter-u-diehl/stdp-mnist

Brian2 version created by Xu Zhang
GitHub updated 2016-09-13
https://github.com/zxzhijia/Brian2STDPMNIST

Python3 version and refactoring by Steven P. Bamford
https://github.com/bamford/Brian2STDPMNIST

This version by Waleed El-Geresy
https://github.com/sdpenguin/Brian2STDPMNIST

@author: Waleed El-Geresy
"""

# conda create -y -n brian2 python=3
# conda install -y -n brian2 -c conda-forge numpy scipy matplotlib brian2 pandas ipython pytables

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
from inspect import currentframe, getframeinfo
import json

from utilities import (
    Config,
    Dataset,
    load_connections,
    save_connections,
    get_initial_weights,
    load_theta,
    save_theta,
    get_matrix_from_file,
    connections_to_file,
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
    create_test_store,
)

from neurons import DiehlAndCookExcitatoryNeuronGroup, DiehlAndCookInhibitoryNeuronGroup
from synapses import DiehlAndCookSynapses

# b2.set_device('cpp_standalone', build_on_run=False)  # cannot use with network operations
# b2.prefs.codegen.target = 'numpy'  # faster startup, but slower iterations

def main(config):
    try:
        os.makedirs(config.run_path, exist_ok=(config.clobber or config.resume or config.test_mode))
    except (OSError, FileExistsError):
        print(f"Refusing to overwrite existing output files in {config.run_path}")
        print(f"Use --clobber to force overwriting")
        exit(8)
    if config.test_mode:
        suffix = "_test"
    else:
        suffix = ""
    if config.resume:
        mode = "a"
    else:
        mode = "w"
    config.logfile_name = os.path.join(config.run_path, f"output{suffix}.log")
    fh = logging.FileHandler(config.logfile_name, mode)
    fh.setLevel(logging.DEBUG if config.debug else logging.INFO)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    store_filename = os.path.join(config.run_path, f"store{suffix}.h5")
    os.makedirs(config.weight_path, exist_ok=True)
    os.makedirs(config.output_path, exist_ok=True)
    if config.test_mode:
        # TODO: MAKE THIS WORK WITH ORIGINAL DC15 WEIGHTS
        create_test_store(store_filename, os.path.join(config.run_path, f"store.h5"))
        mode = "a"
    with pd.HDFStore(store_filename, mode=mode, complib="blosc", complevel=9) as store:
        simulation(config, store)


def simulation(config, store):
    #### Load Training/Testing Data and Apply Configuration###

    dataset = Dataset(config)
    config.add_data_params(dataset)
    data = dataset.data

    #### Initialise Metadata with Configuration ####

    metadata = store.root._v_attrs # Access the store metadata
    for x in config.__dict__: setattr(metadata, x, getattr(config, x))
    if not config.resume:
        metadata.nseen = 0
        metadata.nprogress = 0

    #### Log Configuration ####

    log.info("Brian2STDPMNIST/simulation.py")
    log.info("Arguments =============")
    log.info("{} : {}".format(key, getattr(metadata, key)) for key in metadata)
    log.info("=======================")
    if config.test_mode:
        log.info("Testing run {}".format(config.runname))
    elif config.resume:
        log.info("Resuming training run {}".format(config.runname))
    else:
        log.info("Training run {}".format(config.runname))

    log.info('Original defaultclock.dt = {}'.format(str(b2.defaultclock.dt)))
    b2.defaultclock.dt = config.clock * b2.ms
    metadata["dt"] = b2.defaultclock.dt
    log.info("defaultclock.dt = {}".format(str(b2.defaultclock.dt)))

    #### Create Network Population and Recurrent Connections ####

    initial_weight_matrices = get_initial_weights(config.n_neurons)
    neuron_groups = {}
    connections = {}
    spike_monitors = {}
    state_monitors = {}
    network_operations = []

    for subgroup_n, name in enumerate(config.population_names):
        log.info(f"Creating neuron group {name}")
        subpop_e = name + "e"
        subpop_i = name + "i"
        const_theta = False
        neuron_namespace = {}
        if name == "A" and config.tc_theta is not None:
            neuron_namespace["tc_theta"] = config.tc_theta * b2.ms
        if name == "O":
            neuron_namespace["tc_theta"] = 1e6 * b2.ms
        if config.test_mode:
            const_theta = True
            if name == "O":
                # TODO: move to a config variable
                neuron_namespace["tc_theta"] = 1e5 * b2.ms
                const_theta = False
        nge = neuron_groups[subpop_e] = DiehlAndCookExcitatoryNeuronGroup(
            config.n_neurons[subpop_e],
            const_theta=const_theta,
            timer=config.timer,
            custom_namespace=neuron_namespace,
        )
        ngi = neuron_groups[subpop_i] = DiehlAndCookInhibitoryNeuronGroup(
            config.n_neurons[subpop_i]
        )

        if not config.random_weights:
            theta_saved = load_theta(name, config.weight_path)
            if len(theta_saved) != config.n_neurons[subpop_e]:
                raise ValueError(
                    f"Requested size of neuron population {subpop_e} "
                    f"({n_neurons[subpop_e]}) does not match size of "
                    f"saved data ({len(theta_saved)})"
                )
            neuron_groups[subpop_e].theta = theta_saved
        elif name in config.theta_init:
            neuron_groups[subpop_e].theta = config.theta_init[name]

        for connType in config.recurrent_conntype_names:
            log.info(f"Creating recurrent connections for {connType}")
            preName = name + connType[0]
            postName = name + connType[1]
            connName = preName + postName
            conn = connections[connName] = DiehlAndCookSynapses(
                neuron_groups[preName], neuron_groups[postName], conn_type=connType
            )
            conn.connect()  # all-to-all connection
            minDelay, maxDelay = config.delay[connType]
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
            weightMatrix = None
            if config.use_premade_weights:
                try:
                    weightMatrix = load_connections(connName, config.random_weight_path if config.random_weights else config.weight_path)
                except FileNotFoundError:
                    log.info(
                        f"Requested premade {'random' if config.random_weights else ''} "
                        f"weights, but none found for {connName}"
                    )
            if weightMatrix is None:
                log.info("Using generated initial weight matrices")
                weightMatrix = initial_weight_matrices[connName]
            conn.w = weightMatrix.flatten()

        log.debug(f"Creating spike monitors for {name}")
        spike_monitors[subpop_e] = b2.SpikeMonitor(nge, record=config.record_spikes)
        spike_monitors[subpop_i] = b2.SpikeMonitor(ngi, record=config.record_spikes)
        if config.monitoring:
            log.debug(f"Creating state monitors for {name}")
            state_monitors[subpop_e] = b2.StateMonitor(
                nge,
                variables=True,
                record=range(0, config.n_neurons[subpop_e], 10),
                dt=0.5 * b2.ms,
            )

    if config.test_mode and config.supervised:
        # make output neurons more sensitive
        neuron_groups["Oe"].theta = 5.0 * b2.mV  # TODO: refine

    #### Create TimedArray of Rates for Input Examples ####

    input_rates = np.zeros((config.n_data * config.n_dt_total, config.n_neurons["Xe"]), dtype=np.float16)
    log.info("Preparing input rate stream {}".format(input_rates.shape))
    for j in range(config.n_data):
        spike_rates = data["x"][j].reshape(config.n_neurons["Xe"]) / 8
        spike_rates *= config.input_intensity
        start = j * config.n_dt_total
        input_rates[start : start + config.n_dt_example] = spike_rates
    input_rates = input_rates * b2.Hz
    stimulus_X = b2.TimedArray(input_rates, dt=config.input_dt)
    total_data_time = config.n_data * config.n_dt_total * config.input_dt

    #### Create TimedArray of Rates for Input Labels ####

    if "Y" in config.input_population_names:
        input_label_rates = np.zeros(
            (config.n_data * config.n_dt_total, config.n_neurons["Ye"]), dtype=np.float16
        )
        log.info("Preparing input label rate stream {}".format(input_label_rates.shape))
        if not config.test_mode:
            label_spike_rates = to_categorical(data["y"], dtype=np.float16)
        else:
            label_spike_rates = np.ones(config.n_data)
        label_spike_rates *= config.input_label_intensity
        for j in range(config.n_data):
            start = j * config.n_dt_total
            input_label_rates[start : start + config.n_dt_example] = label_spike_rates[j]
        input_label_rates = input_label_rates * b2.Hz
        stimulus_Y = b2.TimedArray(input_label_rates, dt=config.input_dt)

    #### Create Input Population and Connections from Input Populations ####

    for k, name in enumerate(config.input_population_names):
        subpop_e = name + "e"
        # stimulus is repeated for duration of simulation
        # (i.e. if there are multiple epochs)
        neuron_groups[subpop_e] = b2.PoissonGroup(
            config.n_neurons[subpop_e], rates=f"stimulus_{name}(t % total_data_time, i)"
        )
        log.debug(f"Creating spike monitors for {name}")
        spike_monitors[subpop_e] = b2.SpikeMonitor(
            neuron_groups[subpop_e], record=config.record_spikes
        )

    for name in config.connection_names:
        log.info(f"Creating connections between {name[0]} and {name[1]}")
        for connType in config.forward_conntype_names:
            log.debug(f"connType {connType}")
            preName = name[0] + connType[0]
            postName = name[1] + connType[1]
            connName = preName + postName
            stdp_on = config.ee_STDP_on and connName in config.stdp_conn_names
            nu_factor = 10.0 if name in ["AO"] else None
            conn = connections[connName] = DiehlAndCookSynapses(
                neuron_groups[preName],
                neuron_groups[postName],
                conn_type=connType,
                stdp_on=stdp_on,
                stdp_rule=config.stdp_rule,
                custom_namespace=config.synapse_namespace,
                nu_factor=nu_factor,
            )
            conn.connect()  # all-to-all connection
            minDelay, maxDelay = config.delay[connType]
            if maxDelay > 0:
                deltaDelay = maxDelay - minDelay
                conn.delay = "minDelay + rand() * deltaDelay"
            weightMatrix = None
            if config.use_premade_weights:
                try:
                    weightMatrix = load_connections(connName, config.random_weight_path if config.random_weights else config.weight_path)
                except FileNotFoundError:
                    log.info(
                        f"Requested premade {'random' if config.random_weights else ''} "
                        f"weights, but none found for {connName}"
                    )
            if weightMatrix is None:
                log.info("Using generated initial weight matrices")
                weightMatrix = initial_weight_matrices[connName]
            conn.w = weightMatrix.flatten()
            if config.monitoring:
                log.debug(f"Creating state monitors for {connName}")
                state_monitors[connName] = b2.StateMonitor(
                    conn,
                    variables=True,
                    record=range(0, config.n_neurons[preName] * config.n_neurons[postName], 1000),
                    dt=5 * b2.ms,
                )

    if config.ee_STDP_on:
        @b2.network_operation(dt=config.total_example_time, order=1)
        def normalize_weights(t):
            for connName in connections:
                if connName in config.stdp_conn_names:
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
                    colFactors[ok] = config.total_weight[connName] / colSums[ok]
                    connweights *= colFactors
                    conn.w = connweights.flatten()

        network_operations.append(normalize_weights)

    def record_cumulative_spike_counts(t=None):
        if t is None or t > 0:
            metadata.nseen += 1
        for name in config.population_names + config.input_population_names:
            subpop_e = name + "e"
            count = pd.DataFrame(
                spike_monitors[subpop_e].count[:][None, :], index=[metadata.nseen]
            )
            count = count.rename_axis("tbin")
            count = count.rename_axis("neuron", axis="columns")
            store.append(f"cumulative_spike_counts/{subpop_e}", count)

    @b2.network_operation(dt=config.total_example_time, order=0)
    def record_cumulative_spike_counts_net_op(t):
        record_cumulative_spike_counts(t)

    network_operations.append(record_cumulative_spike_counts_net_op)

    def progress():
        ''' Logs the progress. '''
        log.debug("Starting progress")
        starttime = time.process_time()
        labels = get_labels(data)
        log.info("So far seen {} examples".format(metadata.nseen))
        store.append(
            f"nseen", pd.Series(data=[metadata.nseen], index=[metadata.nprogress])
        )
        metadata.nprogress += 1
        assignments_window, accuracy_window = get_windows(
            metadata.nseen, config.progress_assignments_window, config.progress_accuracy_window
        )
        for name in config.population_names + config.input_population_names:
            log.debug(f"Progress for population {name}")
            subpop_e = name + "e"
            csc = store.select(f"cumulative_spike_counts/{subpop_e}")
            spikecounts_present = spike_counts_from_cumulative(
                csc, config.n_data, metadata.nseen, config.n_neurons[subpop_e], start=-accuracy_window
            )
            n_spikes_present = spikecounts_present["count"].sum()
            if n_spikes_present > 0:
                spikerates = (
                    spikecounts_present.groupby("i")["count"].mean().astype(np.float32)
                )
                # this reindex no longer necessary?
                spikerates = spikerates.reindex(
                    np.arange(config.n_neurons[subpop_e]), fill_value=0
                )
                spikerates = add_nseen_index(spikerates, metadata.nseen)
                store.append(f"rates/{subpop_e}", spikerates)
                store.flush()
                fn = os.path.join(
                    config.output_path, "spikerates-summary-{}.pdf".format(subpop_e)
                )
                plot_rates_summary(
                    store.select(f"rates/{subpop_e}"), filename=fn, label=subpop_e
                )
            if name in config.population_names:
                if not config.test_mode:
                    spikecounts_past = spike_counts_from_cumulative(
                        csc,
                        config.n_data,
                        metadata.nseen,
                        config.n_neurons[subpop_e],
                        end=-accuracy_window,
                        atmost=assignments_window,
                    )
                    n_spikes_past = spikecounts_past["count"].sum()
                    log.debug("Assignments based on {} spikes".format(n_spikes_past))
                    if name == "O":
                        assignments = pd.DataFrame(
                            {"label": np.arange(config.n_neurons[subpop_e], dtype=np.int32)}
                        )
                    else:
                        assignments = get_assignments(spikecounts_past, labels)
                    assignments = add_nseen_index(assignments, metadata.nseen)
                    store.append(f"assignments/{subpop_e}", assignments)
                else:
                    assignments = store.select(f"assignments/{subpop_e}")
                if n_spikes_present == 0:
                    log.debug(
                        "No spikes in present interval - skipping accuracy estimate"
                    )
                else:
                    log.debug("Accuracy based on {} spikes".format(n_spikes_present))
                    predictions = get_predictions(
                        spikecounts_present, assignments, labels
                    )
                    accuracy = get_accuracy(predictions, metadata.nseen)
                    store.append(f"accuracy/{subpop_e}", accuracy)
                    store.flush()
                    accuracy_msg = (
                        "Accuracy [{}]: {:.1f}%  ({:.1f}–{:.1f}% 1σ conf. int.)\n"
                        "{:.1f}% of examples have no prediction\n"
                        "Accuracy excluding non-predictions: "
                        "{:.1f}%  ({:.1f}–{:.1f}% 1σ conf. int.)"
                    )

                    log.info(accuracy_msg.format(subpop_e, *accuracy.values.flat))
                    plot_accuracy(
                        store.select(f"accuracy/{subpop_e}"),
                        filename=os.path.join(config.output_path,
                        "accuracy-{}.pdf".format(subpop_e)
                    ))
                    plot_quantity(
                        spikerates,
                        filename=os.path.join(config.output_path, "spikerates-{}.pdf".format(subpop_e)),
                        label=f"spike rate {subpop_e}",
                        nseen=metadata.nseen,
                    )
                theta = theta_to_pandas(subpop_e, neuron_groups, metadata.nseen)
                store.append(f"theta/{subpop_e}", theta)
                fn = os.path.join(config.output_path, "theta-{}.pdf".format(subpop_e))
                plot_quantity(
                    theta,
                    filename=fn,
                    label=f"theta {subpop_e} (mV)",
                    nseen=metadata.nseen,
                )
                fn = os.path.join(
                    config.output_path, "theta-summary-{}.pdf".format(subpop_e)
                )
                plot_theta_summary(
                    store.select(f"theta/{subpop_e}"), filename=fn, label=subpop_e
                )
        if not config.test_mode or metadata.nseen == 0:
            ''' Save connection weights. '''
            for conn in config.save_conns:
                log.info(f"Saving connection {conn}")
                conn_df = connections_to_pandas(connections[conn], metadata.nseen)
                store.append(f"connections/{conn}", conn_df)
            for conn in config.plot_conns:
                log.info(f"Plotting connection {conn}")
                subpop = conn[-2:]
                if "O" in conn:
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
                    output=("O" in conn),
                    feedback=("O" in conn[:2]),
                    label=conn,
                )
            if config.monitoring:
                for km, vm in spike_monitors.items():
                    states = vm.get_states()
                    with open(
                        os.path.join(
                            config.output_path, f"saved-spikemonitor-{km}.pickle"
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(states, f)

                for km, vm in state_monitors.items():
                    states = vm.get_states()
                    with open(
                        os.path.join(
                            config.output_path, f"saved-statemonitor-{km}.pickle"
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(states, f)

        log.debug("progress() took {:.3f} seconds".format(time.process_time() - starttime))

    if config.progress_interval > 0:
        @b2.network_operation(dt=config.total_example_time * config.progress_interval, order=2)
        def progress_net_op(t):
            # if t < total_example_time:
            #    return None
            progress()

        network_operations.append(progress_net_op)

    #### Run the Simulation and Set Inputs ####

    log.info("Constructing the network")
    net = b2.Network()
    for obj_list in [neuron_groups, connections, spike_monitors, state_monitors]:
        for key in obj_list:
            net.add(obj_list[key])

    for obj in network_operations:
        net.add(obj)

    log.info("Starting simulations")

    net.run(config.runtime, report="text", report_period=(60 * b2.second), profile=config.profile)

    b2.device.build(
        directory=os.path.join(config.run_path, "build"), compile=True, run=True, debug=False
    )

    if config.profile:
        log.debug(b2.profiling_summary(net, 10))

    #### Save Results ####

    log.info("Saving results")
    progress()
    if not config.test_mode:
        record_cumulative_spike_counts()
        save_theta(config.population_names, neuron_groups, config.weight_path)
        save_connections(connections, config.save_conns, config.weight_path)


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
    mode_group.add_argument("--test", dest="test_mode", action="store_true", help="Enable test mode")
    mode_group.add_argument("--train", dest="test_mode", action="store_false", help="Enable train mode")

    parser.add_argument("--runname", type=str, default=None, help="Name of output folder, if none given defaults to date and time.")
    parser.add_argument("--runpath", dest='run_path_parent', type=str, default="~/Data/SNN/Brian2STDPMNIST/runs/", help="Parent path for runs folder.")
    parser.add_argument("--datapath", dest="data_path", type=str, default="~/datasets/mnist", help="Path to store/get the MNIST .npz file.")

    debug_group = parser.add_mutually_exclusive_group(required=False)
    # argparse.SUPPRESS makes debug default to True
    debug_group.add_argument("--debug", dest="debug", action="store_true", default=argparse.SUPPRESS,  help="Include debug output from log file")
    debug_group.add_argument("--no-debug", dest="debug", action="store_false", help="Omit debug output in log file")

    parser.add_argument( "--clobber", action="store_true", help="Force overwrite of files in existing run folder")
    parser.add_argument("--num_epochs", type=float, default=None)
    parser.add_argument("--progress_interval", type=int, default=None,
        help=(
            "The number of examples after which to run the progress() function."
            "This function saves parameters to the store file and logs progress."
        ), # The interval at which to save progress
    )
    parser.add_argument("--assignments_window", type=int, default=None)
    parser.add_argument("--accuracy_window", type=int, default=None)
    parser.add_argument("--record_spikes", action="store_true")
    parser.add_argument( "--monitoring", action="store_true",
        help=(
            "Turn on detailed monitoring of spikes and states. "
            "These are pickled and saved each progress interval. "
            "Use with caution: this greatly slows down the "
            "simulation and vastly increases memory usage."
        ),
    )
    parser.add_argument("--permute_data", action="store_true")
    parser.add_argument( "--size", type=int, default=400,
        help="""Number of neurons in the computational layer.
                Currently this must be a square number.""",
    )
    parser.add_argument( "--resume", action="store_true", help="Continue on from existing run")
    parser.add_argument( "--stdp_rule", type=str, default="original",
        choices=[
            "original",
            "minimal-triplet",
            "full-triplet",
            "powerlaw",
            "exponential",
            "symmetric",
        ],
    )
    parser.add_argument( "--custom_namespace", "--synapse_namespace", dest="custom_namespace", type=str, default="{}",
        help=(
            "Customise the synapse namespace. "
            "This should be given as a dictionary, surrounded by quotes (in JSON format), "
            'for example: \'{"tar": 0.1, "mu": 2.0}\'.' # TODO: Check if this should be 'tau'?
        ),
    )
    parser.add_argument( "--total_input_weight", type=float,
        help=(
            "The total weight of input synapses into each neuron, "
            "enforced by normalisation after each example. "
            "Default is the number of input neurons divided by 10, "
            "which is very close to the DC15 value of 78.0."
        ),
    )
    parser.add_argument("--tc_theta", type=float, help="The theta time constant")
    parser.add_argument("--timer", type=float, help="Modify dtimer/dt for the 'spike suppression timer'. Can be zero to disable timer.")
    parser.add_argument("--use_premade_weights", action="store_true")
    parser.add_argument("--supervised", action="store_true", help="Enable supervised training")
    parser.add_argument("--feedback", action="store_true", help="Enable feedback in supervised training")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--clock", type=float, help="The simulation resolution in milliseconds (default 0.5)")

    parser.add_argument("--dc15", action="store_true", help="Set all options to reproduce DC15 as closely as possible")

    args = parser.parse_args()
    config = Config(args)

    sys.exit(main(config))
