#!/bin/env python
'''
Original Python2/Brian1 version created by Peter U. Diehl
on 2014-12-15, GitHub updated 2015-08-07
https://github.com/peter-u-diehl/stdp-mnist

Brian2 version created by Xu Zhang
GitHub updated 2016-09-13
https://github.com/zxzhijia/Brian2STDPMNIST

This version created by Steven P. Bamford
https://github.com/bamford/Brian2STDPMNIST

@author: Steven P. Bamford
'''

# conda install -c conda-forge numpy scipy matplotlib keras brian2
# conda install -c brian-team brian2tools

import logging
logging.captureWarnings(True)
log = logging.getLogger('spiking-mnist')
log.setLevel(logging.DEBUG)

import matplotlib as mpl
mpl.use('PDF')

import os.path
import numpy as np
import matplotlib.cm as cmap
from scipy import sparse
import brian2 as b2
import brian2tools as b2t
from keras.datasets import mnist


class config:
    # a global object to store configuration info
    pass

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------


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


def save_connections():
    log.info('Saving connections')
    for connName in save_conns:
        conn = connections[connName]
        connListSparse = list(zip(conn.i, conn.j, conn.w))
        out = os.path.join(config.data_path,
                           'weights/{}{}'.format(connName, config.ending))
        np.save(out, connListSparse)


def save_theta():
    print('save theta')
    for pop_name in population_names:
        np.save(config.data_path + 'weights/theta_' + pop_name +
                config.ending, neuron_groups[pop_name + 'e'].theta)


def normalize_weights(connections, weight):
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            conn = connections[connName]
            len_source = len(conn.source)
            len_target = len(conn.target)
            connweights = np.zeros((len_source, len_target))
            connweights[conn.i, conn.j] = conn.w
            colSums = connweights.sum(axis=0)
            colFactors = weight['ee_input'] / colSums
            connweights *= colFactors
            conn.w = connweights[conn.i, conn.j]


def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = np.zeros((n_input, n_e))
    connMatrix[connections[name].i, connections[name].j] = connections[name].w
    weight_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            rearranged_weights[i * n_in_sqrt: (i + 1) * n_in_sqrt, j * n_in_sqrt: (j + 1) * n_in_sqrt] = \
                weight_matrix[:, i + j *
                              n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b2.figure(fig_num, figsize=(18, 18))
    im2 = b2.imshow(weights, interpolation="nearest", vmin=0,
                    vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    b2.colorbar(im2)
    b2.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig


def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num / update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num,
                               0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def plot_performance(fig_num):
    num_evaluations = int(num_examples / update_interval)
    time_steps = list(range(0, num_evaluations))
    performance = np.zeros(num_evaluations)
    fig = b2.figure(fig_num, figsize=(5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance)  # my_cmap
    b2.ylim(ymax=100)
    b2.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig


def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance


def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(
                spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(
                result_monitor[input_nums == j], axis=0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments


def main(test_mode=True):
    # load MNIST
    training, testing = get_labeled_data()

    #-------------------------------------------------------------------------
    # set parameters and equations
    #-------------------------------------------------------------------------

    np.random.seed(0)
    config.data_path = './'
    if test_mode:
        weight_path = config.data_path + 'weights/'
        num_examples = 10000 * 1
        use_testing_set = True
        do_plot_performance = False
        record_spikes = True
        ee_STDP_on = False
        update_interval = num_examples
    else:
        weight_path = config.data_path + 'random/'
        num_examples = 60000 * 3
        use_testing_set = False
        do_plot_performance = True
        if num_examples <= 60000:
            record_spikes = True
        else:
            record_spikes = True
        ee_STDP_on = True

    config.ending = ''
    n_input = 784
    n_e = 400
    n_i = n_e
    single_example_time = 0.35 * b2.second
    resting_time = 0.15 * b2.second
    runtime = num_examples * (single_example_time + resting_time)
    if num_examples <= 10000:
        update_interval = num_examples
        weight_update_interval = 20
    else:
        update_interval = 10000
        weight_update_interval = 100
    if num_examples <= 60000:
        save_connections_interval = 10000
    else:
        save_connections_interval = 10000
        update_interval = 10000

    v_rest_e = -65. * b2.mV
    v_rest_i = -60. * b2.mV
    v_reset_e = -65. * b2.mV
    v_reset_i = -45. * b2.mV
    v_thresh_e = -52. * b2.mV
    v_thresh_i = -40. * b2.mV
    refrac_e = 5. * b2.ms
    refrac_i = 2. * b2.ms

    weight = {}
    delay = {}
    input_population_names = ['X']
    population_names = ['A']
    input_connection_names = ['XA']
    save_conns = ['XeAe']
    input_conn_names = ['ee_input']
    recurrent_conn_names = ['ei', 'ie']
    weight['ee_input'] = 78.
    delay['ee_input'] = (0 * b2.ms, 10 * b2.ms)
    delay['ei_input'] = (0 * b2.ms, 5 * b2.ms)
    input_intensity = 2.
    start_input_intensity = input_intensity

    tc_pre_ee = 20 * b2.ms
    tc_post_1_ee = 20 * b2.ms
    tc_post_2_ee = 40 * b2.ms
    nu_ee_pre = 0.0001      # learning rate
    nu_ee_post = 0.01       # learning rate
    wmax_ee = 1.0
    exp_ee_pre = 0.2
    exp_ee_post = exp_ee_pre
    STDP_offset = 0.4

    if test_mode:
        scr_e = 'v = v_reset_e; timer = 0*ms'
    else:
        tc_theta = 1e7 * b2.ms
        theta_plus_e = 0.05 * b2.mV
        scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
    offset = 20.0 * b2.mV
    v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
    v_thresh_i_str = 'v>v_thresh_i'
    v_reset_i_str = 'v=v_reset_i'

    neuron_eqs_e = '''
            dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-100.*mV-v)                          : amp
            dge/dt = -ge/(1.0*ms)                                   : 1
            dgi/dt = -gi/(2.0*ms)                                  : 1
            '''
    if test_mode:
        neuron_eqs_e += '\n  theta      :volt'
    else:
        neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
    neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

    neuron_eqs_i = '''
            dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-85.*mV-v)                          : amp
            dge/dt = -ge/(1.0*ms)                                   : 1
            dgi/dt = -gi/(2.0*ms)                                  : 1
            '''
    eqs_stdp_ee = '''
                    post2before                            : 1
                    dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                    dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                    dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
                '''
    eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
    eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

    b2.ion()
    fig_num = 1
    neuron_groups = {}
    input_groups = {}
    connections = {}
    rate_monitors = {}
    spike_monitors = {}
    spike_counters = {}
    result_monitor = np.zeros((update_interval, n_e))

    neuron_groups['e'] = b2.NeuronGroup(n_e * len(population_names), neuron_eqs_e,
                                        threshold=v_thresh_e_str, refractory=refrac_e, reset=scr_e, method='euler')
    neuron_groups['i'] = b2.NeuronGroup(n_i * len(population_names), neuron_eqs_i,
                                        threshold=v_thresh_i_str, refractory=refrac_i, reset=v_reset_i_str, method='euler')

    #-------------------------------------------------------------------------
    # create network population and recurrent connections
    #-------------------------------------------------------------------------
    for subgroup_n, name in enumerate(population_names):
        log.info(f'Creating neuron group {name}')

        neuron_groups[name + 'e'] = neuron_groups['e'][subgroup_n *
                                                       n_e:(subgroup_n + 1) * n_e]
        neuron_groups[name + 'i'] = neuron_groups['i'][subgroup_n *
                                                       n_i:(subgroup_n + 1) * n_e]

        neuron_groups[name + 'e'].v = v_rest_e - 40. * b2.mV
        neuron_groups[name + 'i'].v = v_rest_i - 40. * b2.mV
        if test_mode or weight_path[-8:] == 'weights/':
            neuron_groups['e'].theta = np.load(
                weight_path + 'theta_' + name + config.ending + '.npy') * b2.volt
        else:
            neuron_groups['e'].theta = np.ones((n_e)) * 20.0 * b2.mV

        log.info(f'Creating recurrent connections')
        for conn_type in recurrent_conn_names:
            connName = name + conn_type[0] + name + conn_type[1]
            weightMatrix = get_matrix_from_file(
                weight_path + '../random/' + connName + config.ending + '.npy')
            model = 'w : 1'
            pre = 'g%s_post += w' % conn_type[0]
            post = ''
            if ee_STDP_on:
                if 'ee' in recurrent_conn_names:
                    model += eqs_stdp_ee
                    pre += '; ' + eqs_stdp_pre_ee
                    post = eqs_stdp_post_ee
            connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                model=model, on_pre=pre, on_post=post)
            connections[connName].connect()  # all-to-all connection
            connections[connName].w = weightMatrix[connections[connName].i,
                                                   connections[connName].j]

        log.info(f'Creating monitors for {name}')
        rate_monitors[name +
                      'e'] = b2.PopulationRateMonitor(neuron_groups[name + 'e'])
        rate_monitors[name +
                      'i'] = b2.PopulationRateMonitor(neuron_groups[name + 'i'])
        spike_counters[name + 'e'] = b2.SpikeMonitor(neuron_groups[name + 'e'])

        if record_spikes:
            spike_monitors[name +
                           'e'] = b2.SpikeMonitor(neuron_groups[name + 'e'])
            spike_monitors[name +
                           'i'] = b2.SpikeMonitor(neuron_groups[name + 'i'])

    #-------------------------------------------------------------------------
    # create input population and connections from input populations
    #-------------------------------------------------------------------------
    pop_values = [0, 0, 0]
    for i, name in enumerate(input_population_names):
        input_groups[name + 'e'] = b2.PoissonGroup(n_input, 0 * b2.Hz)
        rate_monitors[name +
                      'e'] = b2.PopulationRateMonitor(input_groups[name + 'e'])

    for name in input_connection_names:
        log.info(f'Creating connections between {name[0]} and {name[1]}')
        for connType in input_conn_names:
            log.debug(f'connType {connType} of {input_conn_names}')
            connName = name[0] + connType[0] + name[1] + connType[1]
            weightMatrix = get_matrix_from_file(
                weight_path + connName + config.ending + '.npy')
            model = 'w : 1'
            pre = 'g%s_post += w' % connType[0]
            post = ''
            if ee_STDP_on:
                log.info(f'Creating STDP for connection {name[0]}e{name[1]}e')
                model += eqs_stdp_ee
                pre += '; ' + eqs_stdp_pre_ee
                post = eqs_stdp_post_ee

            connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                model=model, on_pre=pre, on_post=post)
            minDelay = delay[connType][0]
            maxDelay = delay[connType][1]
            deltaDelay = maxDelay - minDelay
            # TODO: test this
            connections[connName].connect(True)  # all-to-all connection
            connections[connName].delay = 'minDelay + rand() * deltaDelay'
            connections[connName].w = weightMatrix[connections[connName].i,
                                                   connections[connName].j]

    #-------------------------------------------------------------------------
    # run the simulation and set inputs
    #-------------------------------------------------------------------------
    log.info('Constructing the network')
    net = b2.Network()
    for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
                     spike_monitors, spike_counters]:
        for key in obj_list:
            net.add(obj_list[key])

    previous_spike_count = np.zeros(n_e)
    assignments = np.zeros(n_e)
    input_numbers = [0] * num_examples
    outputNumbers = np.zeros((num_examples, 10))
    if not test_mode:
        input_weight_monitor, fig_weights = plot_2d_input_weights()
        fig_num += 1
    if do_plot_performance:
        performance_monitor, performance, fig_num, fig_performance = plot_performance(
            fig_num)
    for i, name in enumerate(input_population_names):
        input_groups[name + 'e'].rates = 0 * b2.Hz
    log.info('Starting simulations')
    net.run(0 * b2.second)
    j = 0
    while j < (int(num_examples)):
        if test_mode:
            if use_testing_set:
                spike_rates = testing['x'][j % 10000, :, :].reshape(
                    (n_input)) / 8. * input_intensity
            else:
                spike_rates = training['x'][j % 60000, :, :].reshape(
                    (n_input)) / 8. * input_intensity
        else:
            normalize_weights(connections, weight)
            spike_rates = training['x'][j % 60000, :, :].reshape(
                (n_input)) / 8. * input_intensity
        input_groups['Xe'].rates = spike_rates * b2.Hz
        log.info(f'run number {j+1} of {int(num_examples)}')
        net.run(single_example_time, report=None)

        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(
                result_monitor[:], input_numbers[j - update_interval: j])
        if j % weight_update_interval == 0 and not test_mode:
            update_2d_input_weights(input_weight_monitor, fig_weights)
        if j % save_connections_interval == 0 and j > 0 and not test_mode:
            save_connections(str(j))
            save_theta(str(j))

        current_spike_count = np.asarray(
            spike_counters['Ae'].count[:]) - previous_spike_count
        previous_spike_count = np.copy(spike_counters['Ae'].count[:])
        if np.sum(current_spike_count) < 5:
            input_intensity += 1
            for i, name in enumerate(input_population_names):
                input_groups[name + 'e'].rates = 0 * b2.Hz
            net.run(resting_time)
        else:
            result_monitor[j % update_interval, :] = current_spike_count
            if test_mode and use_testing_set:
                input_numbers[j] = testing['y'][j % 10000]
            else:
                input_numbers[j] = training['y'][j % 60000]
            outputNumbers[j, :] = get_recognized_number_ranking(
                assignments, result_monitor[j % update_interval, :])
            if j % 100 == 0 and j > 0:
                print('runs done:', j, 'of', int(num_examples))
            if j % update_interval == 0 and j > 0:
                if do_plot_performance:
                    unused, performance = update_performance_plot(
                        performance_monitor, performance, j, fig_performance)
                    print('Classification performance',
                          performance[:(j / float(update_interval)) + 1])
            for i, name in enumerate(input_population_names):
                input_groups[name + 'e'].rates = 0 * b2.Hz
            net.run(resting_time)
            input_intensity = start_input_intensity
            j += 1

    #-------------------------------------------------------------------------
    # save results
    #-------------------------------------------------------------------------
    print('save results')
    if not test_mode:
        save_theta()
    if not test_mode:
        save_connections()
    else:
        np.save(config.data_path + 'activity/resultPopVecs' +
                str(num_examples), result_monitor)
        np.save(config.data_path + 'activity/inputNumbers' +
                str(num_examples), input_numbers)

    #-------------------------------------------------------------------------
    # plot results
    #-------------------------------------------------------------------------
    if rate_monitors:
        b2.figure(fig_num)
        fig_num += 1
        for i, name in enumerate(rate_monitors):
            b2.subplot(len(rate_monitors), 1, 1 + i)
            b2.plot(rate_monitors[name].t / b2.second,
                    rate_monitors[name].rate, '.')
            b2.title('Rates of population ' + name)

    if spike_monitors:
        b2.figure(fig_num)
        fig_num += 1
        for i, name in enumerate(spike_monitors):
            b2.subplot(len(spike_monitors), 1, 1 + i)
            b2.plot(spike_monitors[name].t / b2.ms,
                    spike_monitors[name].i, '.')
            b2.title('Spikes of population ' + name)

    if spike_counters:
        b2.figure(fig_num)
        fig_num += 1
        b2.plot(spike_monitors['Ae'].count[:])
        b2.title('Spike count of population Ae')

    plot_2d_input_weights()

    plt.figure(5)

    subplot(3, 1, 1)

    brian_plot(connections['XeAe'].w)
    subplot(3, 1, 2)

    brian_plot(connections['AeAi'].w)

    subplot(3, 1, 3)

    brian_plot(connections['AiAe'].w)

    plt.figure(6)

    subplot(3, 1, 1)

    brian_plot(connections['XeAe'].delay)
    subplot(3, 1, 2)

    brian_plot(connections['AeAi'].delay)

    subplot(3, 1, 3)

    brian_plot(connections['AiAe'].delay)

    b2.ioff()
    b2.show()


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description=('Brian2 implementation of Diehl & Cook 2015, '
                     'an MNIST classifer constructed from a '
                     'Spiking Neural Network with STDP-based learning.'))
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--test', dest='test_mode', action='store_true',
                            help='Enable test mode')
    mode_group.add_argument('--train', dest='test_mode', action='store_false',
                            help='Enable train mode')
    args = parser.parse_args()

    sys.exit(main(test_mode=args.test_mode))
