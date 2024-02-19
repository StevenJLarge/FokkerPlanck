# This is a class-based implementation of a simulator object that can
# run simulations of particular FPE instances. Effectively this acts as
# a convenience wrapper around the raw fpe integraator obects
from typing import Iterable, Optional, Dict, Callable
import numpy as np

from fokker_planck.types.basetypes import CPVector
from fokker_planck.simulator.base import StaticSimulator1D, DynamicSimulator1D
import fokker_planck.forceFunctions as ff


class Protocol:
    def __init__(self, control_parameter_vals: Iterable, time_vals: Iterable):
        self._CP = control_parameter_vals
        self._time = time_vals

    @property
    def CP(self):
        return self._CP

    @property
    def time(self):
        return self._time


# STATIC SIMUlATORS
class HarmonicEquilibrationSimulator(StaticSimulator1D):
    def __init__(
        self, fpe_config: Dict, k_trap: float, trap_min: Optional[float] = 0
    ):
        super().__init__(fpe_config)
        self.force_func = ff.harmonic_force
        self.force_params = [k_trap, trap_min]

    def initialize_probability(self, mean: Optional[float] = None, init_var: Optional[float] = None):
        if init_var is None:
            x_len = len(self.fpe_args.x_array)
            uni_prob = (np.ones(x_len) / ((x_len - 2) * self.fpe_args.dx))
            uni_prob[0] = 0
            uni_prob[-1] = 0
            self.fpe_prob = uni_prob
        else:
            if mean is None:
                mean = self.force_params[1]
            self.fpe.initialize_probability(mean, init_var)


class PeriodicEquilibrationSimulator(StaticSimulator1D):
    def __init__(
        self, fpe_config: Dict, amp: float, n_minima: int,
        phase_shift: float = 0
    ):
        super().__init__(fpe_config)
        self.force_func = ff.periodic_force
        self.force_params = [amp, n_minima, phase_shift]

    def initialize_probability(self, classification: Optional[str] = None, **kwargs):
        if classification is None:
            x_len = len(self.fpe_args.x_array)
            uni_prob = np.ones(x_len) / (x_len * self.fpe_args.dx)
            self.fpe_prob = uni_prob

        if classification.lower() == 'gaussian':
            self.fpe.initialize_probability(**kwargs)

        raise NotImplementedError(
            f'Probability classificaition {classification} not recognized, '
            'currently only None (uniform) and gaussian are supported'
        )


# DYNAMIC SIMULATORS
class BreathingSimulator(DynamicSimulator1D):
    def __init__(
        self, fpe_config: Dict, k_i: float, k_f: float,
        force_function: Callable, energy_function: Callable
    ):
        super().__init__(fpe_config, k_i, k_f)
        self.force_func = force_function
        self.energy_func = energy_function

    def build_friction_array(self) -> np.ndarray:
        return 1 / (self.lambda_array ** 3)

    def initialize_probability(self):
        self.fpe.initialize_probability(0, 1 / self.lambda_init)

    def update(self, protocol_bkw: CPVector, protocol_fwd: CPVector):
        params_bkw = ([protocol_bkw, 0])
        params_fwd = ([protocol_fwd, 0])
        if not self.check_cfl(params_fwd):
            raise ValueError('CFL not satisfied!')

        self.fpe.work_step(
            params_bkw, params_fwd, self.force_func, self.energy_func
        )

    @property
    def total_work(self):
        return self.fpe.work_accumulator

    @property
    def work(self):
        return self.fpe.work_tracker[::self.tracking_stride]

    @property
    def simulation_time(self):
        return self.time_tracker

    @property
    def power(self):
        return self.fpe.power_tracker[::self.tracking_stride]


class HarmonicTranslationSimulator(DynamicSimulator1D):
    def __init__(
        self, fpe_config: Dict, trap_init: float,
        trap_fin: float, trap_strength: float,
        force_function: Optional[Callable] = ff.harmonic_force_const_velocity,
        energy_function: Optional[Callable] = ff.harmonic_energy_const_velocity,
    ):
        super().__init__(fpe_config, trap_init, trap_fin)
        self.force_func = force_function
        self.energy_func = energy_function
        self.trap_strength = trap_strength

    def build_friction_array(self) -> np.ndarray:
        return np.ones_like(self.lambda_array)

    def initialize_probability(self):
        self.fpe.initialize_probability(self.lambda_init, 1 / self.trap_strength)

    def update(self, protocol_bkw: CPVector, protocol_fwd: CPVector):
        dlambda = protocol_fwd - protocol_bkw
        cp_vel = dlambda / self.fpe.dt
        # Params: [k_trap, center, velocity, D]
        params_bkw = ([self.trap_strength, 0, cp_vel, self.fpe.D])
        params_fwd = ([self.trap_strength, dlambda, cp_vel, self.fpe.D])

        if not self.check_cfl(params_fwd, self.force_func):
            raise ValueError('CFL not satisfied!')

        self.fpe.work_step(
            params_bkw, params_fwd, self.force_func, self.energy_func
        )


if __name__ == "__main__":
    # 2 equivalent sample input_config dictionaries
    config_1 = {
        "D": 1.0,
        "dx": 0.01,
        "dt": 0.001,
        "x_min": -3,
        "x_max": 3
    }

    config_2 = {
        "D": 1.0,
        "dx": 0.01,
        "dt": 0.001,
        "x_array": np.arange(-3, 3, 0.01)
    }

    breathing_1 = BreathingSimulator(
        config_1, 0.5, 4.0, ff.harmonic_force, ff.harmonic_energy
    )
    breathing_2 = BreathingSimulator(
        config_2, 0.5, 4.0, ff.harmonic_force, ff.harmonic_energy
    )

    harmonic_trap = HarmonicTranslationSimulator(
        config_1, 0, 5, 8, ff.harmonic_force_const_velocity,
        ff.harmonic_energy_const_velocity
    )

    proto_n_1 = breathing_1.build_protocol(1.5, mode="naive")
    proto_n_2 = breathing_2.build_protocol(1.5, mode="naive")

    proto_o_1 = breathing_1.build_protocol(1.5, mode="optimal")
    proto_o_2 = breathing_2.build_protocol(1.5, mode="optimal")

    proto_n_3 = harmonic_trap.build_protocol(1.5, mode="naive")
    proto_n_4 = harmonic_trap.build_protocol(3.0, mode="naive")

    proto_o_3 = harmonic_trap.build_protocol(1.5, mode="optimal")
    proto_o_4 = harmonic_trap.build_protocol(3.0, mode="optimal")

    print("DONE!")
