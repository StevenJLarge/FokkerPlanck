# This is a class-based implementation of a simulator object that can
# run simulations of particular FPE instances. Effectively this acts as
# a convenience wrapper around the raw fpe integraator obects
from abc import ABCMeta
from typing import Union, Iterable, Optional, Dict
import numpy as np
import scipy.interpolate
import copy
from FPE.Integrator import FPE_Integrator_1D

CPVector = Union[float, Iterable]


class SimulationResult:
    def __init__(self):
        pass


class BaseSimulator(metaclass=ABCMeta):

    def __init__(self, lambda_init: CPVector, lambda_fin: CPVector):        
        self.lambda_init = lambda_init
        self.lambda_fin = lambda_fin

        self.tau = None
        self.time_array = None


class Simulator1D(BaseSimulator):

    self.lambda_init: CPVector
    self.lambda_fin: CPVector
    self.tau: float
    self.time_array: Iterable

    def __init__(
        self, fpe_config: Dict, lambda_init: CPVector, lambda_fin: CPVector,
        n_lambda: Optional[int] = 500
    ):
        fpe_args, fpe_kwargs = self._parseInputConfig(fpe_config)
        D, dt, dx, x_array = fpe_args
        self.fpe = FPE_Integrator_1D(D, dt, dx, x_array, **fpe_kwargs)
        self.lambda_array = np.linspace(lambda_init, lambda_fin, n_lambda)

        super().__init__(lambda_init, lambda_fin)

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

    def _get_raw_velocity(self, friction: float):
        return 1.0 / np.sqrt(friction)

    def _get_raw_times(self) -> np.ndarray:
        raw_times = np.zeros_like(self.lambda_array)
        for i in range(len(raw_times) - 1):
            numerator = 2 * (self.k_array[i+1] - self.k_array[i])
            denominator = (self.raw_velocities[i+1] + self.raw_velocities[i])
            raw_times[i+1] =  (numerator / denominator) + raw_times[i]

        return raw_times

    def _get_real_times(self, raw_times: np.ndarray) -> np.ndarray:
        raw_tau = raw_times[-1]
        self.real_times = (self.tau / raw_tau) * raw_times

    def _get_real_velocities(self, raw_times: np.ndarray) -> np.ndarray:
        raw_tau = raw_times[-1]
        self.real_velocities = (raw_tau / tau) * self.raw_velocities

    def build_optimal_protocol(self, tau: float) -> np.ndarray:
        raw_times = self._get_raw_times()
        self._get_real_times(raw_times)
        self._get_real_velocities(raw_times)

        # Now, we can generate the pairs of (time, k_value) from the pairs of real_times, k_array
        # But we want a uniform set of times at which we know the k values, so we cna interpolate
        time_interp = scipy.interpolate.interp1d(self.real_times, self.k_array, fill_value="extrapolate")
        self.optimal_protocol = time_interp(self.time_array)
        return self.optimal_protocol

    def build_naive_protocol(self, tau: float):
        self.naive_protocol = np.linspace(self.k_i, self.k_f, len(self.time_array))

    def run_simulation(self, mode: Optional[str] = "naive"):
        if mode == "naive":
            self.build_naive_protocol()
            protocol = self.naive_protocol
        elif mode == "optimal":
            self.build_optimal_protocol()
            protocol = self.optimal_protocol
        else:
            raise ValueError("`mode` parameter must be `naive` or `optimal`")

        prob_tracker = []
        time_tracker = []
        cp_tracker = []

        self.fpe.initializeProbability(0, 1 / self.k_i)
        self.fpe.initializePhysicalTrackers()

        for i, k in enumerate(protocol[:-1]):
            if i % 10 == 0:
                prob_tracker.append(self.fpe.prob)
                time_tracker.append(self.time_array[i])
                cp_tracker.append(protocol[i])
            self.fpe.work_step(([protocol[i], 0]), ([protocol[i+1], 0]), ff.harmonicForce, ff.harmonicEnergy)

        return SimulationResult(self, cp_tracker, prob_tracker, time_tracker)


class BreathingSimulator(Simulator1D):
    # instance specific logic: what needs to be provided FOR EACH unique system
    # we are looking at?
    pass



if __name__ == "__main__":
    # 2 equivalent sample input_config dictionaries
    config_1 = {
        "D": 1.0,
        "dx": 0.01,
        "dt": 0.001,
        "x_min": -2,
        "x_max": 2,
    }

    config_2 = {
        "D": 1.0,
        "dx": 0.01,
        "dt": 0.001,
        "x_array": np.arange(-2, 2, 0.001)
    }
