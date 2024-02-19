from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, auto
import numpy as np
import scipy.interpolate
from typing import Dict, Optional, Union, Iterable, Tuple, Callable

from fokker_planck.types.basetypes import CPVector
from fokker_planck.integrator import FokkerPlanck1D
from fokker_planck.base import Integrator


class BaseSimulator(ABC):

    def __init__(self, tracking_stride: Optional[int] = 10):
        self.tau = None
        self.time_array = None
        self.tracking_stride = tracking_stride

        self.force_func: Callable = None
        self.fpe: Integrator = None

    def check_cfl(self, params):
        # if not self.fpe.check_CFL(params, self.force_func):
            # raise ValueError('CFL not satisfied!')
        return self.fpe.check_CFL(params, self.force_func)


class KeyElements1D(Enum):
    D: float = auto()
    dx: float = auto()
    dt: float = auto()
    x_min: float = auto()
    x_max: float = auto()
    x_array: float = auto()

    @classmethod
    def __iter__(cls):
        return iter(cls._member_map_.values())

    @classmethod
    def members(cls):
        return cls._member_names_


class InputParams:
    def __init__(self, params: Optional[Dict] = None):
        if params is not None:
            for k, p in params.items():
                if p is None:
                    continue
                self.__dict__[k] = p

    @property
    def as_dict(self):
        return vars(self)


class Simulator1D(BaseSimulator):

    def __init__(self, fpe_config: Dict, tracking_stride: int = 10):
        super().__init__(tracking_stride=tracking_stride)
        self.key_elems = KeyElements1D
        self.fpe_args, self.fpe_kwargs = self._parse_input_config(fpe_config)
        self.fpe = FokkerPlanck1D(
            **self.fpe_args.as_dict, **self.fpe_kwargs.as_dict
        )

    def reset(self):
        self.fpe.reset()

    def _parse_input_config(
        self, input_config: Dict
    ) -> Tuple[InputParams, InputParams]:
        # Pull of key attributes
        key_elem_dict = defaultdict(lambda: None)
        # for k in key_elements:
        for k in self.key_elems:
            key_elem_dict[k.name] = input_config.get(k.name, None)

        if (
            (key_elem_dict['x_array'] is None)
            and any([key_elem_dict[x] is None for x in ['x_min', 'x_max']])
        ):
            raise ValueError(
                'Must provide either `x_array` or `x_min` AND `x_max` in'
                'input_config dictionary'
            )

        if any([key_elem_dict[x] is None for x in ['D', 'dx', 'dt']]):
            raise ValueError(
                'Must provide keys of `D`, `dx`, and `dt` in input config '
                'dictionary.'
            )

        if key_elem_dict['x_array'] is None:
            x_min, x_max, dx = key_elem_dict['x_min'], key_elem_dict['x_max'], key_elem_dict['dx']
            key_elem_dict['x_array'] = np.arange(x_min, x_max, dx)
            del key_elem_dict['x_min']
            del key_elem_dict['x_max']

        additional_specs = {
            k: v for k, v in input_config.items() if k not in self.key_elems.members()
        }
        return InputParams(key_elem_dict), InputParams(additional_specs)


class StaticSimulator1D(Simulator1D):
    def __init__(self, fpe_config: Dict, tracking_stride: int = 10):
        super().__init__(fpe_config, tracking_stride=tracking_stride)
        self.force_params: Iterable = None
        self.force_func: Callable = None

    def run_simulation(self, tau: float):

        if self.force_params is None or self.force_func is None:
            raise ValueError(
                "Must initialize/provide `force_params` and `force_func` "
                "instance variables in StaticSimulator1D subclasses"
            )

        prob_tracker = []
        time_tracker = []
        time_array = np.arange(0, tau, self.fpe_args.dt)

        self.initialize_probability()
        self.fpe.init_physical_trackers()

        self.check_cfl(self.force_params)

        for i, t in enumerate(time_array):
            if i % self.tracking_stride:
                prob_tracker.append(self.fpe.prob)
                time_tracker.append(t)
            self.fpe.integrate_step(self.force_params, self.force_func)

        return SimulationResult(
            self, prob_tracker, time_tracker, sim_type='static'
        )

    @abstractmethod
    def initialize_probability(self):
        pass


class DynamicSimulator1D(Simulator1D):

    def __init__(
        self, fpe_config: Dict, lambda_init: CPVector, lambda_fin: CPVector,
        n_lambda: Optional[int] = 500, tracking_stride: int = 10
    ):
        super().__init__(fpe_config, tracking_stride=tracking_stride)
        self.lambda_array = np.linspace(lambda_init, lambda_fin, n_lambda)
        self.lambda_init = lambda_init
        self.lambda_fin = lambda_fin

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
        self.fpe.reset()

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

        return SimulationResult(
            self, prob_tracker, time_tracker, sim_type='dynamic',
            cp_tracker=cp_tracker
        )

    def build_friction_array(self) -> np.ndarray:
        raise NotImplementedError(
            'If you want to use optimal protocol mode, you must implement '
            '`build_friction_array` in the derived class of DynamicSimulator1D'
        )

    @abstractmethod
    def initialize_probability(self):
        pass

    @abstractmethod
    def update(self, protocol_bkw: CPVector, protocol_fwd: CPVector):
        pass


class SimulationResult:
    def __init__(
        self, sim: BaseSimulator, prob_tracker: Iterable,
        time_tracker: Iterable, sim_type: str,
        cp_tracker: Optional[Iterable] = None
    ):
        self._sim = sim
        self.x_array = self._sim.fpe.x_array
        self.CP = cp_tracker
        self.prob_tracker = prob_tracker
        self.time = time_tracker
        self.sim_type = sim_type

    # Helper classes?


# Empty class implementation for testing base functionality
class VoidSimulator(DynamicSimulator1D):
    def __init__(self, fpe_config: Dict):
        super().__init__(fpe_config, 0, 1)

    def build_friction_array(self):
        pass

    def initialize_probability(self):
        pass

    def update(self):
        pass
