# This is currently just a file for collecting bits of code that is
# not currently used in simulation.py, etc.

def get_2d_input_weights(connections, blank=False):
    conn = connections['XeAe']
    n_input = len(conn.source)
    n_e = len(conn.target)
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    if not blank:
        weights = np.reshape(conn.w, (n_input, n_e))
        for i in range(n_e_sqrt):
            for j in range(n_e_sqrt):
                wk = weights[:, i + j * n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
                rearranged_weights[i * n_in_sqrt: (i + 1) * n_in_sqrt,
                                   j * n_in_sqrt: (j + 1) * n_in_sqrt] = wk
    return rearranged_weights


def create_2d_input_weights_plot(connections, max_weight=1.0):
    name = 'XeAe'
    weights = get_2d_input_weights(connections, blank=True)
    fig, ax = b2.subplots(figsize=(18, 18))
    monitor = ax.imshow(weights, interpolation="nearest", vmin=0,
                        vmax=max_weight, cmap=cmap.get_cmap('hot_r'))
    b2.colorbar(monitor)
    b2.title('weights of connection' + name)
    fig.set_tight_layout(True)
    return monitor


def update_2d_input_weights_plot(monitor, connections):
    log.info('Updating 2d input weights plot')
    weights = get_2d_input_weights(connections)
    monitor.set_array(weights)
    fig = monitor.axes.figure
    fig.savefig(os.path.join(config.figure_path, 'input_weights.pdf'))


def get_current_performance(pred_ranking, labels):
    prediction = pred_ranking[-config.update_interval:, 0]
    labels = labels[-config.update_interval:]
    correct = prediction == labels
    return 100 * correct.mean()


def create_performance_plot():
    fig, ax = b2.subplots(figsize=(5, 5))
    monitor, = ax.plot([])
    ax.set_xlabel('time step')
    ax.set_ylabel('accuracy')
    ax.set_ylim(top=100)
    ax.set_title('Classification performance')
    fig.set_tight_layout(True)
    return monitor


def update_performance_plot(monitor, current_step, pred_ranking, labels):
    log.info('Updating performance plot')
    current_perf = get_current_performance(pred_ranking, labels)
    timestep, performance = [i.tolist() for i in monitor.get_data()]
    timestep.append(current_perf)
    performance.append(current_perf)
    monitor.set_data(timestep, performance)
    fig = monitor.axes.figure
    fig.savefig(os.path.join(config.figure_path, 'performance.pdf'))
    return performance


def get_predicted_class_ranking(assignments, spike_rates):
    mean_rates = np.zeros(config.num_classes)
    for i in range(config.num_classes):
        num_assignments = (assignments == i).sum()
        if num_assignments > 0:
            mean_rates[i] = spike_rates[assignments == i].mean()
    return np.argsort(mean_rates)[::-1]


def get_new_assignments(result_monitor, input_labels):
    input_labels = np.asarray(input_labels)
    n_e = result_monitor.shape[1]
    # average rates over all examples for each class
    rates = np.zeros((config.num_classes, n_e))
    for j in range(config.num_classes):
        num_labels = (input_labels == j).sum()
        if num_labels > 0:
            rates[j] = np.mean(result_monitor[input_labels == j], axis=0)
    # assign each neuron to the class producing the highest average rate
    assignments = rates.argmax(axis=1)
    return assignments

#-------------------------------------------------------------------------
    # plot results
    #-------------------------------------------------------------------------
    log.info('Plotting results')
    fig = b2.figure(figsize=(5, 10))
    for i, name in enumerate(rate_monitors):
        b2.subplot(len(rate_monitors), 1, 1 + i)
        t = np.asarray(rate_monitors[name].t)
        rate = rate_monitors[name].smooth_rate(width=0.1 * b2.second)
        sample = max(int(len(t) / 1000), 1)
        b2.plot(t[::sample], rate[::sample], '-')
        b2.title('Rates of population ' + name)
    fig.set_tight_layout(True)
    b2.savefig(os.path.join(config.figure_path, 'rates.pdf'))

    fig = b2.figure()
    for i, name in enumerate(spike_monitors):
        b2.subplot(len(spike_monitors), 1, 1 + i)
        t = np.asarray(spike_monitors[name].t / b2.ms)
        idx = spike_monitors[name].i
        while len(t) > 1000:
            t = t[::10]
            idx = idx[::10]
        b2.plot(t, idx, '.')
        b2.title('Spikes of population ' + name)
    fig.set_tight_layout(True)
    b2.savefig(os.path.join(config.figure_path, 'spikes.pdf'))


    fig = b2.figure()
    b2.plot(spike_monitors['Ae'].count[:])
    b2.title('Spike count of population Ae')
    fig.set_tight_layout(True)
    b2.savefig(os.path.join(config.figure_path, 'counts.pdf'))

    input_weight_plot = create_2d_input_weights_plot(connections)
    update_2d_input_weights_plot(input_weight_plot, connections)

    fig = b2.figure(figsize=(5, 10))
    b2.subplot(3, 1, 1)
    b2t.brian_plot(connections['XeAe'].w)
    b2.subplot(3, 1, 2)
    b2t.brian_plot(connections['AeAi'].w)
    b2.subplot(3, 1, 3)
    b2t.brian_plot(connections['AiAe'].w)
    fig.set_tight_layout(True)
    b2.savefig(os.path.join(config.figure_path, 'connections.pdf'))
