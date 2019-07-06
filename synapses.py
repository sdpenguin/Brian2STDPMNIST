import brian2 as b2


class DiehlAndCookSynapses(b2.Synapses):
    def __init__(self, pre_neuron_group, post_neuron_group, conn_type, stdp_on=False):
        self.pre_conn_type = conn_type[0]
        self.post_conn_type = conn_type[1]
        self.stdp_on = stdp_on
        self.create_namespace()
        self.create_equations()
        super().__init__(
            pre_neuron_group,
            post_neuron_group,
            model=self.model,
            on_pre=self.pre_eqn,
            on_post=self.post_eqn,
            namespace=self.namespace,
        )

    def create_namespace(self):
        self.namespace = {
            "tc_pre_ee": 20 * b2.ms,
            "tc_post_1_ee": 20 * b2.ms,
            "tc_post_2_ee": 40 * b2.ms,
            "nu_ee_pre": 0.0001,
            "nu_ee_post": 0.01,
            "wmax_ee": 1.0,
        }

    def create_equations(self):
        self.model = b2.Equations("w : 1")
        self.pre_eqn = "g{}_post += w".format(self.pre_conn_type)
        self.post_eqn = ""
        if self.stdp_on:
            self.model += b2.Equations(
                """
                post2before  : 1
                dpre/dt = -pre/(tc_pre_ee)  : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)  : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)  : 1 (event-driven)
                """
            )
            self.pre_eqn += """
                pre = 1.
                w = clip(w + nu_ee_pre * post1, 0, wmax_ee)
                """
            self.post_eqn = """
                post2before = post2
                w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee)
                post1 = 1.
                post2 = 1.
                """
