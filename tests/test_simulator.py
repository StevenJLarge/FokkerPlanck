# testing methods for the simulator class
import pytest
import numpy as np

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


if __name__ == "__main__":
    pytest.main([__file__])
