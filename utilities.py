import logging

logging.captureWarnings(True)
log = logging.getLogger("spiking-mnist")

import os.path
import datetime
from math import ceil
import numpy as np
from scipy import sparse
from scipy.special import betaincinv
import brian2 as b2
from urllib.request import urlretrieve
from inspect import getargvalues

import json
import matplotlib

matplotlib.use("PDF")
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits import axes_grid1
import matplotlib.patheffects as path_effects

import pandas as pd

from IPython import embed

class Config(object):
    ''' Configuration parameters for the run.
        Essentially a dictionary that allows attribute access, with autocompletion. '''

    # User defined
    resume = None # Resume a previous run
    runname = None
    run_path_parent = None
    data_path = None
    random_weight_path = None # Location to store random data
    clobber = None
    debug = False
    logfile_name = None
    monitoring = False
    test_mode = None
    feedback = False
    supervised = False
    profile = None
    # Logging:
    save_conns = ["XeAe"] # Connections to save
    plot_conns = ["XeAe"] # Connections to plot
    # Meta parameters
    num_epochs = None
    # Neuron parameters
    latent_dim = 400 # Size of the neuron population (hidden)
    theta_init = {} # O will be initialised to 15mV for supervised training, otherwise nothing
    # Synapse parameters
    delay = {
        "ee" : (0 * b2.ms, 10 * b2.ms),
        "ei" : (0 * b2.ms, 5 * b2.ms),
        "ei_rec" : (0 * b2.ms, 0 * b2.ms),
        "ie_rec" : (0 * b2.ms, 0 * b2.ms)
    }
    stdp_rule = None
    synapse_namespace = None
    total_input_weight = 78.0
    use_premade_weights = True
    ee_STDP_on = None
    input_population_names = ["X"]
    population_names = ["A"]
    connection_names = ["XA"]
    forward_conntype_names = ["ee"]
    recurrent_conntype_names = ["ei_rec", "ie_rec"]
    stdp_conn_names = ["XeAe"]
    total_weight = {}
    input_intensity = 2.0
    input_label_intensity = None
    # Times
    clock = 0.5
    timer = 10.0
    single_example_time = 0.35 * b2.second # Time to simulate a single example
    resting_time = 0.15 * b2.second # Time to wait in between simulations
    runtime = None
    input_dt = 50 * b2.ms
    n_dt_example = None # Number of dt in an example
    # Data parameters
    permute_data = False # Randomly shuffle the training or testing data
    classes = None # MNIST data classes
    num_classes = None
    n_input = None # Shape of the input
    n_data = None # Number of data points
    n_dt_rest = None
    n_dt_total = None
    num_examples = None # Number of examples to train or test on (n_data*num_epochs)
    # Other
    # TODO: Other defaults hidden in the main file
    # TODO: Initialisation parameters for the weights

    neuron_namespaces = {
        "A" : {"tc_theta" : 1.0e7 * b2.ms},
        "O" : {"tc_theta" : 1e6 * b2.ms}
    }
    all_neuron_params = {
        "A" : {"const_theta" : False},
        "O" : {"const_theta" : False}
    }

    def __init__(self, passed_args=None):
        ''' Initialise the configuration object with properties.
            Inputs:
            - passed_args (dict)
        '''
        self.update(passed_args)

    def __str__(self):
        ''' Print the attributes as a dictionary when printed. '''
        return str(self.__dict__)

    def update(self, passed_args):
        ''' Update relevant parameters with values in passed_args.
            Inputs:
            - passed_args (dict)
        '''
        for key in vars(passed_args):
            if getattr(passed_args, key) is None:
                continue
            else:
                setattr(self, key, getattr(passed_args, key))
        self.sanitise_args()

    def sanitise_args(self):
        ''' Ensure that dependent arguments are consistent.
            You must run this manually currently.
            TODO: Find a method for automatically updating this whenever
            relevant parameters are changed. '''
        self.custom_namespace = json.loads(self.custom_namespace.replace("'", '"'))
        if self.runname is None: # Note - this will not run the second time
            if self.resume:
                print(f"Must provide runname to resume")
                exit(2)
            else:
                self.runname = datetime.datetime.now().replace(microsecond=0).isoformat()
        #### Define directory names ####
        self.data_path = os.path.expanduser(self.data_path)
        self.run_path_parent = os.path.expanduser(self.run_path_parent)
        self.run_path = os.path.join(self.run_path_parent, self.runname)
        # Random weights can be fixed across multiple runs, so can be stored alongside persistent files, like the data
        self.random_weight_path = os.path.join(self.data_path, "random/")
        self.output_path = os.path.join(self.run_path, "output{}".format('_test' if self.test_mode else ''))
        self.weight_path = os.path.join(self.run_path, "weights/")

        # Configure test and training dependent parameters
        if self.test_mode:
            self.use_premade_weights = True
            self.ee_STDP_on = False
            self.num_epochs = 1
            self.progress_interval = 1000
            self.progress_assignments_window = 0
            self.progress_accuracy_window = 1000000
            self.input_label_intensity = 0.0
            self.neuron_namespaces["O"]["tc_theta"] = 1e-5 * b2.ms # TODO: Why?
            self.all_neuron_params["A"]["const_theta"] = True # TODO: Why?
        else:
            self.ee_STDP_on = True
            self.num_epochs = 3
            self.progress_interval = 1000
            self.progress_assignments_window = 1000
            self.progress_accuracy_window = 1000
            self.input_label_intensity = 10.0

        self.total_example_time = self.single_example_time + self.resting_time # Total time for an example
        self.n_dt_example = int(round(self.single_example_time / self.input_dt))

        self.n_dt_rest = int(round(self.resting_time / self.input_dt))
        self.n_dt_total = int(self.n_dt_example + self.n_dt_rest)

    def add_data_params(self, dataset):
        ''' Update the configuration with the Dataset dependent parameters. '''
        self.num_classes = dataset.num_classes
        self.n_input = dataset.n_input
        self.n_data = dataset.n_data
        self.num_examples = dataset.num_examples

        self.n_neurons = {
            "Ae": self.latent_dim,
            "Ai": self.latent_dim,
            "Oe": self.num_classes,
            "Oi": self.num_classes,
            "Xe": self.n_input,
            "Ye": self.num_classes,
        }

        if self.total_input_weight is None:
            self.total_weight["XeAe"] = self.n_neurons["Xe"] / 10.0  # standard dc15 value was 78.0
        else:
            self.total_weight["XeAe"] = self.total_input_weight

        self.runtime = self.num_examples * self.total_example_time

        if self.supervised:
            self.input_population_names += ["Y"]
            self.population_names += ["O"]
            self.connection_names += ["YO", "AO"]
            self.save_conns += ["YeOe", "AeOe"]
            self.plot_conns += ["AeOe"]
            self.stdp_conn_names += ["AeOe"]
            self.total_weight["AeOe"] = self.n_neurons["Ae"] / 5.0  # TODO: refine?
            self.theta_init["O"] = 15.0 * b2.mV

        if self.feedback:
            self.connection_names += ["OA"]
            self.save_conns += ["OeAe"]
            self.plot_conns += ["OeAe"]
            self.stdp_conn_names += ["OeAe"]
            self.total_weight["OeAe"] = self.n_neurons["Oe"] / 5.0  # TODO: refine?


#### Load Numpy Data ####

class Dataset(object):

    data = None # Relevant data, either training or test
    testing = None
    training = None
    classes = None # Classes present in the data
    num_classes = None

    n_input = None # Shape of the input
    n_data = None # Number of data points
    num_examples = None # Number of examples to train or test on (n_data*num_epochs)

    def __init__(self, config):
        training, testing = self.get_labeled_data(config.data_path)
        self.classes = np.unique(training["y"])
        self.num_classes = len(self.classes)
        if config.test_mode:
            self.data = testing
        else:
            self.data = training

        if config.permute_data:
            np.random.seed(0) # For consistency
            sample = np.random.permutation(len(config.data["y"]))
            self.data["x"] = self.data["x"][sample]
            self.data["y"] = self.data["y"][sample]

        self.n_input = self.data["x"][0].size
        self.n_data = self.data["y"].size
        self.num_examples = int(self.n_data) * config.num_epochs

        if config.num_epochs < 1: # If training for only a portion of the dataset
            self.n_data = int(np.ceil(self.n_data * config.num_epochs))
            self.data["x"] = self.data["x"][:self.n_data]
            self.data["y"] = self.data["y"][:self.n_data]

    def get_labeled_data(self, data_path):
        log.info("Loading MNIST data")
        origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        filename = "mnist.npz"
        localfilename = os.path.join(data_path, filename)
        if not os.path.exists(localfilename):
            os.makedirs(data_path, exist_ok=True)
            urlretrieve(origin_folder + filename, localfilename)
        with np.load(localfilename) as f:
            training = {"x": f["x_train"], "y": f["y_train"]}
            testing = {"x": f["x_test"], "y": f["y_test"]}
        return training, testing

#### Load and Save Connection Weights ####

def load_connections(connName, weight_path):
    ''' Load random connections from the random weight file.
        Inputs:
        connName: Name of the connection to load
        weight_path: Directory containing .npy file. '''
    filename = os.path.join(weight_path, "{}.npy".format(connName))
    return get_matrix_from_file(filename)

def get_matrix_from_file(filename, shape=None):
    ''' Return connections as a matrix. Helper function used to load connections.
        filename: File containing the connections
        shape: Shape of the matrix to return TODO: Check this'''
    log.debug(f"Reading matrix from {filename}")
    i, j, data = np.load(filename).T
    i = i.astype(np.int)
    j = j.astype(np.int)
    log.debug(f"Read {len(data)} connections")
    arr = sparse.coo_matrix((data, (i, j)), shape).todense()
    log.debug(f"Created a matrix with shape {arr.shape}")
    # log.debug("Statistics:\n" + pd.Series(arr.flat).describe().to_string())
    return arr

def save_connections(connections, save_conns, weight_path, iteration=None):
    ''' Save connections to a weight file.
        Inputs:
        connections: Object containing connections
        save_conns: Names of connections to save
        weight_path: Directory to save connections to as .npy
        iteration: Optional integer suffix for filename '''
    for connName in save_conns:
        log.info("Saving connections {}".format(connName))
        conn = connections[connName]
        filename = os.path.join(weight_path, "{}".format(connName))
        if iteration is not None:
            filename += "-{:06d}".format(iteration)
        connections_to_file(conn, filename)

def connections_to_file(conn, filename):
    ''' Make a list of the connections and weights and save to the file.
        Inputs:
        conn: Connections with parameters i, j, w
        filename: Path of file to save to '''
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save(filename, connListSparse)

def connections_to_pandas(conn, nseen):
    ''' Converts a connection object to a Pandas Dataframe.
        conn: Connection object in format i,j,w
        nseen: TODO: find out what this is'''
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

def get_connections(n, group_in, group_out, weighting=None, type="invalid"):
    ''' Generate a connection matrix according to the shapes of group_in and group_out,
        as well as weighting, and type.
        Inputs:
        n - dictionary associating names with sizes of neuron groups
        group_in - Pre-synaptic neuron group
        group_out - Post-synaptic neuron group
        weighting - Weighting (shape depends on type)
        type - Type of connection generation method
        Output:
        Matrix generated according to the specifications. '''
    npr = np.random.RandomState(9728364)
    choices = ["ei", "ie", "uniform", "constant", "random"]
    if type not in choices:
        raise ValueError('Please supply a type among {}'.format(choices))
    if not weighting:
        raise ValueError('Please supply a weighting for connection type {}'.format(type))
    if type == choices[0]:
        return np.eye(n[group_in]) * weighting
    elif type == choices[1]:
        return weighting * (1 - np.eye(n[group_in]))
    elif type == choices[2]:
        if len(weighting) != 2:
            raise ValueError('Type "{}", requires a high and a low weight for Uniform randomisation.'.format(choices[2]))
        return npr.uniform(low=weighting[0], high=weighting[1], size=(n[group_in], n[group_out])) # Uniformly sampled connections from the input to the hidden layer.
    elif type == choices[3]:
        return np.zeros((n[group_in], n[group_out])) + weighting
    elif type == choices[4]:
        if len(weighting) != 3:
            raise ValueError('Type "{}", requires a low and heigh weight, and a percentage for random connection.'.format(choices[2]))
        matrix = np.zeros((n[group_in], n[group_out]))
        n_connect = int(weighting[2] * n[group_in] * n[group_out])
        connect = npr.choice(n[group_in] * n[group_out], n_connect, replace=False) # Choose weighting% of connections
        matrix.flat[connect] = npr.uniform(low=weighting[0], high=weighting[1], size=n_connect)
        return matrix

def get_initial_weights(n):
    ''' Initialise weights and connections.
        Inputs:
        n - Dictionary containing numbers of neurons
        Returns:
        matrices - dictionary pointing to numpy arrays defining weights.
        '''
    matrices = {}
    #### Unsupervised Learning: ####
    # This weight is set so that an Ae spike guarantees a corresponding Ai spike
    matrices["AeAi"] = get_connections(n, 'Ae', 'Ai', 10.4, type='ei')
    matrices["AiAe"] = get_connections(n, 'Ai', 'Ae', 17.0, type='ei')
    matrices["XeAe"] = get_connections(n, 'Xe', 'Ae', [0.003, 0.303], type='uniform')
    # XeAi connections not currently used but this is how they appear to be generated from inspection of pre-made weights supplied with DC15 code
    matrices["XeAi"] = get_connections(n, "Xe", "Ai", [0, 0.2, 0.1], type="random")
    #### Supervised Learning: ####
    # for neuron group O --- TODO: refine
    matrices["OeOi"] = get_connections(n, 'Oe', 'Oi', 10.4, type='ei')
    matrices["OiOe"] = get_connections(n, 'Oi', 'Oe', 17.0, type='ie')
    matrices["YeOe"] = get_connections(n, 'Ye', 'Oe', 10.4, type='ei')
    # between neuron groups A and O --- TODO: refine
    # matrices["AeOe"] = get_connections(n, 'Ae', 'Oe', [0.003, 0.303], type="uniform")
    matrices["AeOe"] = get_connections(n, 'Ae', 'Oe', 0.1, type="constant")
    matrices["OeAe"] = get_connections(n, 'Oe', 'Ae', 0.1, type="constant")
    return matrices

#### Load and Save Theta Values ####

def load_theta(population_name, weight_path):
    log.info("Loading theta for population {}".format(population_name))
    filename = os.path.join(weight_path, "theta_{}.npy".format(population_name))
    return np.load(filename) * b2.volt

def save_theta(population_names, neuron_groups, weight_path, iteration=None):
    log.info("Saving theta")
    for pop_name in population_names:
        filename = os.path.join(weight_path, "theta_{}".format(pop_name))
        if iteration is not None:
            filename += "-{:06d}".format(iteration)
        np.save(filename, neuron_groups[pop_name + "e"].theta)

def theta_to_pandas(subpop, neuron_groups, nseen):
    ''' Converts the theta values of ``subpop`` from ``neuron_groups`` to a Pandas Series.
        neuron_groups: A list of groups of neurons
        subpop: The index of the subpopulation selected among ``neuron_groups``
        nseen: An index TODO: find out what this is
        TODO: Check the above. '''
    t = pd.Series(neuron_groups[subpop].theta[:] / b2.mV, dtype=np.float32)
    t = add_nseen_index(t, nseen)
    return t

def get_theta(name, config):
    ''' Loads theta for a neuron subpopulation according to the population name.
        Inputs:
        name - Name of the neuron group
        config - The config object.
        Outputs:
        Loaded theta values. '''
    if config.resume or config.test_mode:
        theta_saved = load_theta(name, config.weight_path)
        if len(theta_saved) != config.n_neurons[name]:
            raise ValueError(
                f"Requested size of neuron population {name} "
                f"({config.n_neurons[name]}) does not match size of "
                f"saved data ({len(theta_saved)})"
            )
        return theta_saved
    elif name in config.theta_init: # Initialise Theta to fresh values if necessary
        return config.theta_init[name]
    else:
        return None

#### Rearrange Weights ####

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

#### Plotting ####

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
    ''' Returns the number of examples for the progress assignment window. '''
    log.debug("Requested assignments window: {}".format(progress_assignments_window))
    log.debug("Requested accuracy window: {}".format(progress_accuracy_window))
    progress_window = progress_assignments_window + progress_accuracy_window
    if progress_window > nseen:
        log.debug("Fewer examples have been seen than required for the requested progress windows.")
        if progress_assignments_window > 0:
            log.debug("Discarding first 20% of available examples to avoid initial contamination.") # TODO: Why?
            log.debug("Dividing remaining 80% in proportion to requested windows.")
            assignments_window = int(0.8 * nseen * progress_assignments_window / progress_window)
            accuracy_window = int(0.8 * nseen * progress_accuracy_window / progress_window)
        else:
            # if requested assignments window is zero, then implies test mode: use all seen
            assignments_window = 0
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


def create_test_store(store_file_name, original_store_file_name):
    if not os.path.exists(original_store_file_name):
        raise ValueError('File {} does not exist. Please try training first.'.format(original_store_file_name))
    with pd.HDFStore(
        original_store_file_name, mode="r", complib="blosc", complevel=9
    ) as originalstore:
        with pd.HDFStore(
            store_file_name, mode="w", complib="blosc", complevel=9
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
