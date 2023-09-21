from typing import Union, Iterable
from enum import Enum, auto


class DiffScheme(Enum):
    CrankNicolson = auto()
    Implicit = auto()
    Explicit = auto()


class BoundaryCondition(Enum):
    HardWall = auto()
    Periodic = auto()
    Open = auto()


class SplitMethod(Enum):
    Lie = auto()
    Strang = auto()
    SymStrang = auto()


CPVector = Union[float, Iterable]
