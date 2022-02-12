# Samples of the FPE Integrator in action

import os
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from FPE import Integrator
import FPE.forceFunctions as ff

from pathlib import Path


def calcDiffusion(BC: Optional[str] = "hard-wall") -> Tuple[List, List]:
    dt = 0.005
    dx = 0.001
    D = 1.0
    xArray = np.arange(-1, 1, dx)

    obj = Integrator.FPE_Integrator_1D(D, dt, dx, xArray, boundaryCond=BC)
    obj.initializeProbability(0, 0.125)

    elapsed_time = 0.0
    counter = 0
    density_tracker = [obj.get_prob]
    time_tracker = [elapsed_time]

    while elapsed_time <= 1.0:
        obj.integrate_step((0,), ff.noForce)
        if counter % 10 == 0:
            print("--")
            density_tracker.append(obj.get_prob)
            time_tracker.append(elapsed_time)
        elapsed_time += dt
        counter += 1

    return density_tracker, time_tracker, xArray


def genDiffusionOnlyPlot(
    density_tracker: List, time_tracker: List, xArray: np.ndarray,
    write_name: str, write_path: str, write_format: Optional[str] = "pdf"
):
    fig, ax = plt.subplots(1, 1, figsize=(6.3, 3.5))
    Pal = sns.color_palette("Spectral", len(density_tracker))
    sns.set(style="darkgrid")

    for i, prob in enumerate(density_tracker):
        ax.plot(xArray, prob, linewidth=2.5, color=Pal[i])

    ax.set_xlabel(r"Position $X$", fontsize=15)
    ax.set_ylabel(r"Probability Density", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(write_path, write_name), format=write_format)
    plt.show()
    plt.close()


if __name__ == "__main__":
    proj_dir = Path().resolve().parents[1]
    write_dir = os.path.join(proj_dir, "results/visualizations/examples")
    write_name = "diffusion_example_HW.pdf"

    density_tracker, time_tracker, xVals = calcDiffusion()
    genDiffusionOnlyPlot(
        density_tracker, time_tracker, xVals, write_name, write_dir
    )
