import logging

logging.captureWarnings(True)
log = logging.getLogger("spiking-mnist")

import brian2 as b2


class DiehlAndCookSynapses(b2.Synapses):
    """Simple model of a synapse between two excitatory neurons with STDP"""

    def __init__(
        self,
        pre_neuron_group,
        post_neuron_group,
        conn_type,
        stdp_on=False,
        stdp_rule="original",
        custom_namespace=None,
        nu_factor=None,
    ):
        self.pre_conn_type = conn_type[0]
        self.post_conn_type = conn_type[1]
        self.stdp_rule = stdp_rule
        self.namespace = {}
        self.create_equations()
        if stdp_on:
            self.create_stdp_namespace()
            self.create_stdp_equations()
        if nu_factor is not None:
            for k in self.namespace:
                if "nu" in k:
                    self.namespace[k] *= nu_factor
        if custom_namespace is not None:
            self.namespace.update(custom_namespace)
        log.debug(f"Synapse namespace:\n{self.namespace}".replace(",", ",\n"))
        super().__init__(
            pre_neuron_group,
            post_neuron_group,
            model=self.model,
            on_pre=self.pre_eqn,
            on_post=self.post_eqn,
            namespace=self.namespace,
        )

    def create_equations(self):
        self.model = b2.Equations("w : 1")
        self.pre_eqn = "g{}_post += w".format(self.pre_conn_type)
        self.post_eqn = ""

    def create_stdp_namespace(self):
        if self.stdp_rule == "original":
            self.namespace = {
                "tc_pre_ee": 20 * b2.ms,
                "tc_post_1_ee": 20 * b2.ms,
                "tc_post_2_ee": 40 * b2.ms,
                "nu_ee_pre": 0.0001,
                "nu_ee_post": 0.01,
                "wmax_ee": 1.0,
            }
        elif self.stdp_rule == "minimal-triplet":
            # use values corresponding to DC15 model
            # which approximate those from PG06
            self.namespace = {
                "tc_pre": 20 * b2.ms,
                "tc_post1": 20 * b2.ms,
                "tc_post2": 40 * b2.ms,
                "nu_pair_pre": 0.0001,
                "nu_triple_post": 0.01,
                "wmax": 1.0,
            }
        elif self.stdp_rule == "full-triplet":
            # these values taken from nearest-spike, visual-cortex model of PG06
            self.namespace = {
                "tc_pre1": 16.8 * b2.ms,
                "tc_pre2": 714 * b2.ms,
                "tc_post1": 33.7 * b2.ms,
                "tc_post2": 40 * b2.ms,
                "nu_pair_pre": 6.6e-3,
                "nu_triple_pre": 3.1e-3,
                "nu_pair_post": 8.8e-11,
                "nu_triple_post": 5.3e-2,
                "wmax": 1.0,
            }
        elif self.stdp_rule == "powerlaw":
            self.namespace = {
                "tc_pre": 20 * b2.ms,
                "nu": 0.01,
                "wmax": 1.0,
                "tar": 0.5,  # complete guess!
                "mu": 3.0,  # complete guess!
            }
        elif self.stdp_rule == "exponential":
            self.namespace = {
                "tc_pre": 20 * b2.ms,
                "nu": 0.01,
                "wmax": 1.0,
                "tar": 0.5,  # complete guess!
                "beta": 3.0,  # from Querlioz et al. (2013, doi:10.1109/TNANO.2013.2250995)
            }
        elif self.stdp_rule == "symmetric":
            self.namespace = {
                "tc_pre": 20 * b2.ms,
                "tc_post": 20 * b2.ms,
                "nu_pre": 0.0001,
                "nu_post": 0.01,
                "wmax": 1.0,
                "tar": 0.5,  # complete guess!
                "mu": 3.0,  # complete guess!
            }

    def create_stdp_equations(self):
        if self.stdp_rule == "original":
            # original code from Diehl & Cooke (2015) repository
            # An implementation of the nearest-spike minimal triplet
            # model of Pfister & Gerstner (2006)
            # NOTES:
            # * the sign of the weight pre-synaptic weight update
            #   appears to be wrong compared to PG06
            self.model += b2.Equations(
                """
                dpre/dt = -pre/(tc_pre_ee)  : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)  : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)  : 1 (event-driven)
                """
            )
            self.pre_eqn += """
                pre = 1.
                w = clip(w + nu_ee_pre * post1, 0, wmax_ee)
                """
            self.post_eqn += """
                w = clip(w + nu_ee_post * pre * post2, 0, wmax_ee)
                post1 = 1.
                post2 = 1.
                """
        elif self.stdp_rule == "minimal-triplet":
            # Minimal (visual cortex) model of Pfister & Gerstner (2006)
            # Mapping of notation to PG06 and DC15:
            # pre = r_1 = pre
            # tc_pre = \tau_+ = tc_pre_ee
            # post1 = o_1 = post1
            # post2 = o_2 = post2
            # tc_post1 = \tau_- = tc_post_1_ee
            # tc_post2 = \tau_y = tc_post_2_ee
            # nu_pair_pre = A^-_2 = nu_ee_pre
            # nu_triple_post = A^+_3 = nu_ee_post
            self.model += b2.Equations(
                """
                dpre/dt = -pre / tc_pre  : 1 (event-driven)
                dpost1/dt  = -post1 / tc_post1  : 1 (event-driven)
                dpost2/dt  = -post2 / tc_post2  : 1 (event-driven)
                """
            )
            self.pre_eqn += """
                w = clip(w - post1 * nu_pair_pre, 0, wmax)
                pre = 1.0
                """
            self.post_eqn += """
                w = clip(w + pre * nu_triple_post * post2, 0, wmax)
                post1 = 1.0
                post2 = 1.0
                """
        elif self.stdp_rule == "full-triplet":
            # Full model of Pfister & Gerstner (2006)
            # Mapping of notation to PG06 and DC15:
            # pre1 = r_1 = pre
            # pre2 = r_2  (neglected in minimal model)
            # tc_pre1 = \tau_+ = tc_pre_ee
            # tc_pre2 = \tau_x  (neglected in minimal model)
            # post1 = o_1 = post1
            # post2 = o_2 = post2
            # tc_post1 = \tau_- = tc_post_1_ee
            # tc_post2 = \tau_y = tc_post_2_ee
            # nu_pair_pre = A^-_2 = nu_ee_pre
            # nu_triple_pre = A^-_3  (neglected in minimal model)
            # nu_pair_post = A^+_2  (neglected in minimal model)
            # nu_triple_post = A^+_3 = nu_ee_post
            self.model += b2.Equations(
                """
                dpre1/dt = -pre1 / tc_pre1  : 1 (event-driven)
                dpre2/dt = -pre2 / tc_pre2  : 1 (event-driven)
                dpost1/dt  = -post1 / tc_post1  : 1 (event-driven)
                dpost2/dt  = -post2 / tc_post2  : 1 (event-driven)
                """
            )
            self.pre_eqn += """
                w = clip(w - post1 * (nu_pair_pre + nu_triple_pre * pre2), 0, wmax)
                pre1 = 1.0
                pre2 = 1.0
                """
            self.post_eqn += """
                w = clip(w + pre1 * (nu_pair_post + nu_triple_post * post2), 0, wmax)
                post1 = 1.0
                post2 = 1.0
                """
        if self.stdp_rule == "powerlaw":
            # inferred code from Diehl & Cooke (2015)
            self.model += b2.Equations(
                """
                dpre/dt = -pre/ tc_pre  : 1 (event-driven)
                """
            )
            self.pre_eqn += """
                pre = pre + 1.0
                """
            self.post_eqn += """
                w = clip(w + nu * (pre - tar) * (wmax - w)**mu, 0, wmax)
                """
        if self.stdp_rule == "exponential":
            # inferred code from Diehl & Cooke (2015)
            self.model += b2.Equations(
                """
                dpre/dt = -pre/ tc_pre  : 1 (event-driven)
                """
            )
            self.pre_eqn += """
                pre = pre + 1.0
                """
            self.post_eqn += """
                w = clip(w + nu * (pre * exp(-beta * w) - tar * exp(-beta * (wmax - w))), 0, wmax)
                """
        if self.stdp_rule == "symmetric":
            # inferred code from Diehl & Cooke (2015)
            self.model += b2.Equations(
                """
                dpre/dt = -pre/ tc_pre  : 1 (event-driven)
                dpost/dt  = -post / tc_post  : 1 (event-driven)

                """
            )
            self.pre_eqn += """
                 w = clip(w - nu_pre * pre * w**mu, 0, wmax)
                 pre = pre + 1.0
                 """
            self.post_eqn += """
                 w = clip(w + nu_post * (pre - tar) * (wmax - w)**mu, 0, wmax)
                 post = post + 1.0
                 """
        if self.stdp_rule == "clopath2010":
            # TODO: try Clopath et al. 2010 rule
            # not in DC15, but would be nice to try this sometime:
            # spike-timing dependent plasticity perhaps more bio-physically
            # mediated by the post-synaptic membrane voltage
            pass
