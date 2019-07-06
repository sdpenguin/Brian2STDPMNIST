import brian2 as b2


class DiehlAndCookBaseNeuronGroup(b2.NeuronGroup):
    def __init__(self):
        self.create_namespace()
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


class DiehlAndCookExcitatoryNeuronGroup(DiehlAndCookBaseNeuronGroup):
    def __init__(self, N, test_mode=True):
        self.N = N
        self.test_mode = test_mode
        super().__init__()

    def create_namespace(self):
        self.namespace = {
            "v_rest_e": -65.0 * b2.mV,
            "v_reset_e": -65.0 * b2.mV,
            "v_thresh_e": -52.0 * b2.mV,
            "theta_plus_e": 0.05 * b2.mV,
            "theta_init": 20.0 * b2.mV,
            "refrac_e": 5.0 * b2.ms,
            "tc_theta": 1e7 * b2.ms,
        }

    def create_equations(self):
        self.model = b2.Equations(
            """
            dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100*ms)  : volt (unless refractory)
            I_synE = ge * nS * -v  : amp
            I_synI = gi * nS * (-100*mV - v)  : amp
            dge/dt = -ge / (1.0*ms)  : 1
            dgi/dt = -gi / (2.0*ms)  : 1
            dtimer/dt = 0.1  : second
            wtot  : 1
            """
        )
        if self.test_mode:
            self.model += b2.Equations("theta  : volt")
        else:
            self.model += b2.Equations("dtheta/dt = -theta / tc_theta  : volt")

        self.threshold = "(v > (theta - theta_init + v_thresh_e)) and (timer > refrac_e)"

        self.refractory = "refrac_e"

        self.reset = "v = v_reset_e; timer = 0*ms"
        if not self.test_mode:
            self.reset += "; theta += theta_plus_e"

    def initialize(self):
        self.v = self.namespace["v_rest_e"] - 40.0 * b2.mV
        self.theta = self.namespace["theta_init"]


class DiehlAndCookInhibitoryNeuronGroup(DiehlAndCookBaseNeuronGroup):
    def __init__(self, N):
        self.N = N
        self.create_equations()
        super().__init__()

    def create_namespace(self):
        self.namespace = {
            "v_rest_i": -60.0 * b2.mV,
            "v_reset_i": -45.0 * b2.mV,
            "v_thresh_i": -40.0 * b2.mV,
            "refrac_i": 2.0 * b2.ms,
        }

    def create_equations(self):
        self.model = b2.Equations(
            """
            dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
            I_synE = ge * nS * -v  : amp
            I_synI = gi * nS * (-85.*mV-v)  : amp
            dge/dt = -ge/(1.0*ms)  : 1
            dgi/dt = -gi/(2.0*ms)  : 1
            """
        )

        self.threshold = "v > v_thresh_i"

        self.refractory = "refrac_i"

        self.reset = "v = v_reset_i"

    def initialize(self):
        self.v = self.namespace["v_rest_i"] - 40.0 * b2.mV
