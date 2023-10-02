from abc import ABC, abstractmethod
import numpy as np
import scipy.interpolate
from typing import Dict, Optional, Union, Iterable

from fokker_planck.types.basetypes import CPVector
from fokker_planck.Integrator import FokkerPlank1D


class BaseSimulator(ABC):

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
        fpe_args, fpe_kwargs = self._parse_input_config(fpe_config)
        D, dx, dt, x_array = fpe_args
        self.fpe = FokkerPlank1D(D, dt, dx, x_array, **fpe_kwargs)
        self.lambda_array = np.linspace(lambda_init, lambda_fin, n_lambda)

        super().__init__(lambda_init, lambda_fin)

    def reset(self):
        self.fpe.reset()

    def _parse_input_config(self, input_config: Dict):
        # Key elements
        key_elements = ["D", "dx", "dt", "x_array", "x_min", "x_max"]

        # Pull of key attributes
        D = input_config.get("D", None)
        dx = input_config.get("dx", None)
        dt = input_config.get("dt", None)

        # x-array determiniation values
        x_array = input_config.get("x_array", None)
        x_max = input_config.get('x_max', None)
        x_min = input_config.get("x_min", None)

        if (x_array is None) and np.array([x is None for x in [x_min, x_max]]).any():
            raise ValueError(
                'Must provide either `x_array` or `x_min` AND `x_max` in'
                'input_config dictionary'
            )

        if np.array([entry is None for entry in [D, dx, dt]]).any():
            raise ValueError(
                'Must provide keys of `D`, `dx`, and `dt` in input config '
                'dictionary.'
            )

        if x_array is None:
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
        return (tau / raw_tau) * raw_times

    def _get_real_velocities(
        self, raw_times: np.ndarray, raw_velocities: np.ndarray, tau: float
    ) -> np.ndarray:
        raw_tau = raw_times[-1]
        return (raw_tau / tau) * raw_velocities

    def _build_optimal_protocol(self, tau: float) -> np.ndarray:
        raw_velocity = self._get_raw_velocity(self.build_friction_array())
        raw_times = self._get_raw_times(raw_velocity)
        self.real_time = self._get_real_times(raw_times, tau)
        self.real_velocity = self._get_real_velocities(raw_times, raw_velocity, tau)

        # Now, we can generate the pairs of (time, k_value) from the pairs of real_times, k_array
        # But we want a uniform set of times at which we know the k values, so we cna interpolate
        time_interp = scipy.interpolate.interp1d(self.real_time, self.lambda_array, fill_value="extrapolate")
        self.optimal_protocol = time_interp(self.time_array)

    def _build_naive_protocol(self):
        self.naive_protocol = np.linspace(self.lambda_init, self.lambda_fin, len(self.time_array))

    def build_protocol(self, tau: float, mode: Optional[str] = "naive"):
        n_steps = int(tau / self.fpe.dt)
        self.time_array = np.linspace(0, tau, n_steps)

        if mode == "naive":
            self._build_naive_protocol()
            protocol = self.naive_protocol
        elif mode == "optimal":
            self._build_optimal_protocol(tau)
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
        self.fpe.init_physical_trackers()

        for i, p in enumerate(protocol[:-1]):
            if i % 10 == 0:
                prob_tracker.append(self.fpe.prob)
                time_tracker.append(self.time_array[i])
                cp_tracker.append(p)

            self.update(p, protocol[i+1])

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


class SimulationResult:
    def __init__(
        self, sim: BaseSimulator, cp_tracker: Iterable, prob_tracker: Iterable,
        time_tracker: Iterable
    ):
        self._sim = sim
        self.CP = cp_tracker
        self.prob_tracker = prob_tracker
        self.time = time_tracker


# Empty class implementation for testing base functionality
class VoidSimulator(Simulator1D):
    def __init__(self, fpe_config: Dict):
        super().__init__(fpe_config, 0, 1)

    def build_friction_array(self):
        pass

    def initialize_probability(self):
        pass

    def update(self):
        pass
