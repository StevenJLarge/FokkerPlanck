import pytest
import numpy as np

from FPE.Integrator import FPE_Integrator_1D

def test_advection_error_with_invalid_method():
    valid_method = "lax-wendroff"



