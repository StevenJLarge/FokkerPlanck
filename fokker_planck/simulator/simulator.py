# This is a class-based implementation of a simulator object that can
# run simulations of particular FPE instances. Effectively this acts as
# a convenience wrapper around the raw fpe integraator obects
from typing import Iterable, Optional, Dict, Callable
import numpy as np

from fokker_planck.types.basetypes import CPVector
from fokker_planck.simulator.base import Simulator1D
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


class BreathingSimulator(Simulator1D):
    def __init__(
        self, fpe_config: Dict, k_i: float, k_f: float,
        forceFunction: Callable, energyFunction: Callable
    ):
        super().__init__(fpe_config, k_i, k_f)
        self.forceFunc = forceFunction
        self.energyFunc = energyFunction

    def build_friction_array(self) -> np.ndarray:
        return self.lambda_array ** (3/2)

    def initialize_probability(self):
        self.fpe.initialize_probability(0, 1 / self.lambda_init)

    def update(self, protocol_bkw: CPVector, protocol_fwd: CPVector):
        params_bkw = ([protocol_bkw, 0])
        params_fwd = ([protocol_fwd, 0])

        self.fpe.work_step(
            params_bkw, params_fwd, self.forceFunc, self.energyFunc
        )


class HarmonicTranslationSimulator(Simulator1D):
    def __init__(
        self, fpe_config: Dict, trap_init: float,
        trap_fin: float, trap_strength: float,
        forceFunction: Optional[Callable] = ff.harmonic_force_const_velocity,
        energyFunction: Optional[Callable] = ff.harmonic_energy_const_velocity,
    ):
        super().__init__(fpe_config, trap_init, trap_fin)
        self.forceFunc = forceFunction
        self.energyFunc = energyFunction
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

        self.fpe.work_step(
            params_bkw, params_fwd, self.forceFunc, self.energyFunc
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