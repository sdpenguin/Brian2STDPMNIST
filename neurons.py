import logging

logging.captureWarnings(True)
log = logging.getLogger("spiking-mnist")

import brian2 as b2


class DiehlAndCookBaseNeuronGroup(b2.NeuronGroup):
    def __init__(self):
        self.create_base_namespace()
        self.create_namespace()
        self.create_base_equations()
        self.create_equations()
        self.method = "euler"
        super().__init__(
            self.N,
            model=self.model,
            threshold=self.threshold,
            refractory=self.refractory,
            reset=self.reset,
            method=self.method,
            namespace=self.namespace,
        )
        self.initialize()

    def create_base_namespace(self):
        self.namespace = {
            "v_eqm_synE": 0 * b2.mV,
            "tau_ge": 1.0 * b2.ms,
            "tau_gi": 2.0 * b2.ms,
        }

    def create_base_equations(self):
        """v : membrane potential of each neuron
                - in the absence of synaptic currents (`I_syn?`), this decays
                  exponentially to the resting potential (`v_rest_e`) with
                  time constant `tau_leak_e`
           I_synE: current due to excitatory synapses
                - this is proportional to the conductance of the excitatory
                  synapses (`ge`) and the difference between the membrane
                  potential (`v`) and the equilibrium potential of the
                  excitatory synapses (v_eqm_e)
                - as implemented this implicitly includes a constant factor
                  of the membrane resistance
           I_synI: current due to inhibitory synapses
           ge : conductance of excitatory synapses
                - this decays with time constant `tau_ge`
           gi : conductance of inhibitory synapses
                - this decays with time constant `tau_gi`
           v_eqm_synI_e: equilibrium potentials of inhibitory synapses in
                         an excitatory neuron
           v_eqm_synI_i: equilibrium potentials of inhibitory synapses in
                         an inhibitory neuron
           tau_e: time constant for membrane potential in an excitatory neuron
           tau_i: time constant for membrane potential in an inhibitory neuron
           tau_ge: conductance time constant for an excitatory synapse
           tau_gi: conductance time constant for an inhibitory synapse
        """
        self.model = b2.Equations(
            """
            dv/dt = ((v_rest - v) + (I_synE + I_synI) / nS) / tau  : volt (unless refractory)
            I_synE = ge * nS * (v_eqm_synE - v)  : amp
            I_synI = gi * nS * (v_eqm_synI - v)  : amp
            dge/dt = -ge / tau_ge  : 1
            dgi/dt = -gi / tau_gi  : 1
            """
        )


class DiehlAndCookExcitatoryNeuronGroup(DiehlAndCookBaseNeuronGroup):
    """Simple model of an excitatory (pyramidal) neuron"""

    def __init__(self, N, const_theta=True, timer=0.1, custom_namespace=None):
        self.N = N
        self.const_theta = const_theta
        self.timer = timer
        super().__init__()
        if custom_namespace is not None:
            self.namespace.update(custom_namespace)
        log.debug(f"Neuron namespace:\n{self.namespace}".replace(",", ",\n"))

    def create_namespace(self):
        self.namespace.update(
            {
                "v_rest_e": -65.0 * b2.mV,
                "v_reset_e": -65.0 * b2.mV,
                "v_thresh_e": -52.0 * b2.mV,
                "tau_e": 100.0 * b2.ms,
                "v_eqm_synI_e": -100.0 * b2.mV,
                "theta_plus_e": 0.05 * b2.mV,
                "theta_init": 20.0 * b2.mV,
                "refrac_e": 5.0 * b2.ms,
                "tc_theta": 1e7 * b2.ms,
            }
        )

    def create_equations(self):
        if self.timer is not None:
            self.model += b2.Equations(
                f"""
                dtimer/dt = {self.timer:f}  : second
                """
            )
            # This timer seems a bit odd: it increases more slowly than the regular
            # simulation time, and is only used in the threshold code, to prevent spikes.
            # It effectively increase the refractory time affecting spikes (but not dv/dt)
            # by a factor of 10 (to biologically unrealistic values).
            # TODO: should investigate effect of removing extended refractory period
        self.model = b2.Equations(
            str(self.model), v_rest="v_rest_e", tau="tau_e", v_eqm_synI="v_eqm_synI_e"
        )
        if self.const_theta:
            self.model += b2.Equations("theta  : volt")
        else:
            self.model += b2.Equations("dtheta/dt = -theta / tc_theta  : volt")

        self.threshold = "v > (theta - theta_init + v_thresh_e)"
        if self.timer is not None:
            self.threshold = f"({self.threshold}) and (timer > refrac_e)"

        self.refractory = "refrac_e"

        self.reset = "v = v_reset_e"
        if self.timer is not None:
            self.reset += "; timer = 0*ms"
        if not self.const_theta:
            self.reset += "; theta += theta_plus_e"

    def initialize(self):
        # I am not sure why this value is chosen for the initial membrane potential
        self.v = self.namespace["v_rest_e"] - 40.0 * b2.mV
        self.theta = self.namespace["theta_init"]


class DiehlAndCookInhibitoryNeuronGroup(DiehlAndCookBaseNeuronGroup):
    """Simple model of an inhibitory (basket) neuron"""

    def __init__(self, N):
        self.N = N
        super().__init__()

    def create_namespace(self):
        self.namespace.update(
            {
                "v_rest_i": -60.0 * b2.mV,
                "v_reset_i": -45.0 * b2.mV,
                "v_thresh_i": -40.0 * b2.mV,
                "tau_i": 10.0 * b2.ms,
                "v_eqm_synI_i": -85.0 * b2.mV,
                "refrac_i": 2.0 * b2.ms,
            }
        )

    def create_equations(self):
        self.model = b2.Equations(
            str(self.model), v_rest="v_rest_i", tau="tau_i", v_eqm_synI="v_eqm_synI_i"
        )

        self.threshold = "v > v_thresh_i"

        self.refractory = "refrac_i"

        self.reset = "v = v_reset_i"

    def initialize(self):
        self.v = self.namespace["v_rest_i"] - 40.0 * b2.mV
