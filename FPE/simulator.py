# This is a class-based implementation of a simulator object that can
# run simulations of particular FPE instances. Effectively this acts as
# a convenience wrapper around the raw fpe integraator obects
from abc import ABCMeta, abstractmethod
from typing import Union, Iterable, Optional, Dict, Callable
import numpy as np
import scipy.interpolate
from FPE.Integrator import FPE_Integrator_1D
import FPE.forceFunctions as ff

CPVector = Union[float, Iterable]


class SimulationResult:
    def __init__(
        self, cp_tracker: Iterable, prob_tracker: Iterable,
        time_tracker: Iterable
    ):
        self.CP = cp_tracker
        self.prob_tracker = prob_tracker
        self.time = time_tracker


class BaseSimulator(metaclass=ABCMeta):

    def __init__(self, lambda_init: CPVector, lambda_fin: CPVector):
        self.lambda_init = lambda_init
        self.lambda_fin = lambda_fin

        self.tau = None
        self.time_array = None


class Simulator1D(BaseSimulator):

    def __init__(
        self, fpe_config: Dict, lambda_init: CPVector, lambda_fin: CPVector,
        n_lambda: Optional[int] = 500
    ):
        fpe_args, fpe_kwargs = self._parseInputConfig(fpe_config)
        D, dt, dx, x_array = fpe_args
        self.fpe = FPE_Integrator_1D(D, dt, dx, x_array, **fpe_kwargs)
        self.lambda_array = np.linspace(lambda_init, lambda_fin, n_lambda)

        super().__init__(lambda_init, lambda_fin)

    def reset(self):
        self.fpe.reset()

    def _parseinputConfig(input_config: Dict):
        # Key elements
        key_elements = ["D", "dx", "dt", "x_array", "x_min" "x_max"]

        # Pull of key attributes
        D = input_config.get("D", None)
        dx = input_config.get("dx", None)
        dt = input_config.get("dt", None)

        # x-array determiniation values
        x_array = input_config.get("x_array", None)
        x_max = input_config.get('x_max', None)
        x_min = input_config.get("x_min", None)

        if not x_array and not (x_min and x_max):
            raise ValueError(
                'Must provide either `x_array` or `x_min` AND `x_max` in'
                'input_config dictionary'
            )

        if np.array(entry is None for entry in [D, dx, dt]).any():
            raise ValueError(
                'Must provide keys of `D`, `dx`, and `dt` in input config'
                'dictionary.'
            )

        if not x_array:
            x_array = np.arange(x_min, x_max, dx)

        additional_specs = {
            k: v for k, v in input_config.items() if k not in key_elements
        }
        return (D, dx, dt, x_array), additional_specs

    def _get_raw_velocity(self, friction: Union[float, Iterable]):
        return 1.0 / np.sqrt(friction)

    def _get_raw_times(self, raw_velocity: np.ndarray) -> np.ndarray:
        raw_times = np.zeros_like(self.lambda_array)
        for i in range(len(raw_times) - 1):
            numerator = 2 * (self.lambda_array[i+1] - self.lambda_array[i])
            denominator = (raw_velocity[i+1] + raw_velocity[i])
            raw_times[i+1] = (numerator / denominator) + raw_times[i]

        return raw_times

    def _get_real_times(self, raw_times: np.ndarray, tau: float) -> np.ndarray:
        raw_tau = raw_times[-1]
        self.real_times = (tau / raw_tau) * raw_times

    def _get_real_velocities(
        self, raw_times: np.ndarray, tau: float
    ) -> np.ndarray:
        raw_tau = raw_times[-1]
        self.real_velocities = (raw_tau / tau) * self.raw_velocities

    def _build_optimal_protocol(self, tau: float) -> np.ndarray:
        raw_velocity = self._get_raw_velocity(self.build_friction_array())
        raw_times = self._get_raw_times(raw_velocity)
        self.real_time = self._get_real_times(raw_times, tau)
        self.real_velocity = self._get_real_velocities(raw_times, tau)

        # Now, we can generate the pairs of (time, k_value) from the pairs of real_times, k_array
        # But we want a uniform set of times at which we know the k values, so we cna interpolate
        time_interp = scipy.interpolate.interp1d(self.real_time, self.lambda_array, fill_value="extrapolate")
        optimal_protocol = time_interp(self.time_array)
        return optimal_protocol

    def _build_naive_protocol(self):
        self.naive_protocol = np.linspace(self.lambda_init, self.lambda_fin, len(self.time_array))

    def build_protocol(self, tau: float, mode: Optional[str] = "naive"):
        n_steps = int(tau / self.fpe.dt)
        self.time_array = np.linspace(0, tau, n_steps)

        if mode == "naive":
            self._build_naive_protocol()
            protocol = self.naive_protocol
        elif mode == "optimal":
            self._build_optimal_protocol()
            protocol = self.optimal_protocol
        else:
            raise ValueError("`mode` parameter must be `naive` or `optimal`")

        return protocol, self.time_array

    def run_simulation(self, tau: float, mode: Optional[str] = "naive"):

        protocol, _ = self.build_protocol(tau, mode=mode)

        prob_tracker = []
        time_tracker = []
        cp_tracker = []

        self.initialize_probability()
        self.fpe.initializePhysicalTrackers()

        for i in enumerate(protocol[:-1]):
            if i % 10 == 0:
                prob_tracker.append(self.fpe.prob)
                time_tracker.append(self.time_array[i])
                cp_tracker.append(protocol[i])

            self.update(protocol[i], protocol[i+1])

        return SimulationResult(self, cp_tracker, prob_tracker, time_tracker)

    @abstractmethod
    def build_friction_array(self) -> np.ndarray:
        ...

    @abstractmethod
    def initialize_probability(self):
        ...

    @abstractmethod
    def update(self, protocol_bkw: CPVector, protocol_fwd: CPVector):
        ...


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
        self.fpe.initializeProbability(0, 1 / self.lambda_init)

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
        forceFunction: Optional[Callable] = ff.harmonicForce_constVel,
        energyFunction: Optional[Callable] = ff.harmonicEnergy_constVel,
    ):
        super().__init__(fpe_config, trap_init, trap_fin)
        self.forceFunc = forceFunction
        self.energyFunc = energyFunction
        self.trap_strength = trap_strength

    def build_friction_array(self) -> np.ndarray:
        return np.ones_like(self.lambda_array)

    def initialize_probability(self):
        self.fpe.initializeProbability(self.lambda_init, 1 / self.trap_strength)

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
        "x_array": np.arange(-3, 3, 0.001)
    }

    breathing_1 = BreathingSimulator(
        config_1, 0.5, 4.0, ff.harmonicForce, ff.harmonicEnergy
    )
    breathing_2 = BreathingSimulator(
        config_2, 0.5, 4.0, ff.harmonicForce, ff.harmonicEnergy
    )

    harmonic_trap = HarmonicTranslationSimulator(
        config_1, 0, 5, 8, ff.harmonicForce_constVel,
        ff.harmonicEnergy_constVel
    )

    proto_n_1 = breathing_1.build_protocol(1.5, mode="naive")
    proto_n_2 = breathing_2.build_protocol(1.5, mode="naive")

    proto_o_1 = breathing_1.build_protocol(1.5, mode="optimal")
    proto_o_2 = breathing_2.build_protocol(1.5, mode="optimal")

    proto_n_3 = harmonic_trap.build_protocol(1.5, mode="naive")
    proto_n_4 = harmonic_trap.build_protocol(3.0, mode="naive")

    proto_o_3 = harmonic_trap.build_protocol(1.5, mode="optimal")
    proto_o_4 = harmonic_trap.build_protocol(3.0, mode="optimal")
