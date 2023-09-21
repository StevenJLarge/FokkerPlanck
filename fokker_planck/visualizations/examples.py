import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Iterable

from fokker_planck.util import notebooks


# SECTION - Helper routines
def _configure_visualization_format():
    sns.set(style="darkgrid")


def _check_input_file_specs(write_name: str, write_path: str) -> bool:
    if write_name is not None or write_path is not None:
        if write_name is None or write_path is None:
            raise ValueError(
                "You must supply BOTH `write_path` and `write_name`"
            )
        return True
    else:
        return False

# !SECTION

# SECTION - General routines (multi-notebook)


def density_tracking_plot(
    density_tracker: List, time_tracker: List, norm_tracker: List,
    x_array: np.ndarray, write_name: str, write_path: str,
    write_format: Optional[str] = "pdf",
) -> plt.Axes:

    save_file = _check_input_file_specs(write_name, write_path)
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.5))
    Pal = sns.color_palette("Spectral", len(density_tracker))

    for i, prob in enumerate(density_tracker):
        ax[0].plot(x_array, prob, linewidth=2.5, color=Pal[i])

    ax[1].plot(time_tracker, norm_tracker, linewidth=2.5, color='k')

    ax[0].set_xlabel(r"Position $x$", fontsize=15)
    ax[0].set_ylabel(r"$p(x)$", fontsize=15)

    ax[1].set_xlabel(r"Time $t$", fontsize=15)
    ax[1].set_ylabel(r"$\sum p(x_i)\Delta x_i$", fontsize=15)

    ax[1].set_ylim([0, 1.1])

    plt.tight_layout()
    if save_file:
        plt.savefig(os.path.join(write_path, write_name), format=write_format)
    return fig

# !SECTION

# SECTION - Notebook 1 routines - Advection


# !SECTION

# SECTION - Notebook 2 routines - Diffusion
# !SECTION

# SECTION - Notebook 3 routines - Advection Diffusion - relaxation in harmonic BCs

def mean_tracking_plot_harmonic(
    trap_strength: float, D: float, mean_tracker: Iterable[float],
    time_tracker: Iterable[float], write_name: Optional[str] = None,
    write_path: Optional[str] = None, write_format: Optional[str] = "pdf",
) -> plt.Axes:
    _configure_visualization_format()
    save_file = _check_input_file_specs(write_name, write_path)

    fig, ax = plt.subplots(1, 1, figsize=(6.3, 3.5))
    
    time_theory = np.linspace(0, time_tracker[-1], 100)
    relax_theory = np.exp(-trap_strength * time_theory / D)

    mean_tracker /= mean_tracker[0]
    ax.fill_between(time_theory, relax_theory, color=sns.xkcd_rgb["tomato"], alpha=0.2)
    ax.plot(
        time_theory, relax_theory, linewidth=2.5,
        color=sns.xkcd_rgb["tomato"], label="Theory"
    )
    ax.plot(
        time_tracker, mean_tracker, 'o',
        markersize=7, color=sns.xkcd_rgb["electric blue"],
        alpha=0.8, label="Simulation"
    )

    ax.legend(fontsize=12)
    ax.set_xlabel(r"Elapsed Time", fontsize=15)
    ax.set_ylabel(r"Distribution mean $\langle x\rangle$", fontsize=15)
    plt.tight_layout()
    if save_file:
        plt.savefig(os.path.join(write_path, write_name), format=write_format)

    return fig


def harmonic_work_plot(
    work_arr: Iterable, power_arr: Iterable, time: Iterable, vel_arr: Iterable,
    vel_labels: Iterable, trap_strength: float,
    write_dir: Optional[str] = None, write_name: Optional[str] = None
) -> plt.Axes:
    _configure_visualization_format()
    save_file = _check_input_file_specs

    fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.5))
    Pal = sns.color_palette("husl", len(vel_arr))

    theory_arr_work = [
        notebooks.calc_harmonic_power_theory(v, trap_strength, time)
        for v in vel_arr
    ]

    theory_arr_power = [
        notebooks.calc_harmonic_power_theory(v, trap_strength, time)
        for v in vel_arr
    ]

    # Generate work plots
    for i, (w, w_t) in enumerate(zip(work_arr, theory_arr_work)):
        ax[0].plot(
            time, w_t, '--', linewidth=1.0, color='k', alpha=0.6,
            label=(lambda x: "Theory" if x == 0 else None)(i)
        )
        ax[0].plot(time, w, linewidth=2.0, color=Pal[i], label=vel_labels[i])

    # Generate power plots
    for i, (p, p_t) in enumerate(zip(power_arr, theory_arr_power)):
        ax[1].plot(time, p, linewidth=2.0, color=Pal[i])
        ax[1].plot(time, p_t, '--', linewidth=1.0, color='k', alpha=0.6)

    ax[0].legend(fontsize=12)
    ax[0].set_xlabel(r"Elapsed time $\tau$", fontsize=15)
    ax[0].set_ylabel(r"Mean work $\langle W\rangle_{[0, \tau]}$", fontsize=15)
    ax[1].set_xlabel(r"Elapsed time $\tau$", fontsize=15)
    ax[1].set_ylabel(r"Mean power input $\langle P(\tau)\rangle$", fontsize=15)

    plt.tight_layout()
    if save_file:
        plt.savefig(os.path.join(write_dir, write_name), format="pdf")
    return fig


# !SECTION


# SECTION - Notebook 4 routines - Excess work and Power - Theory and numerical
# !SECTION

# SECTION Notebook 5 routines - Periodic system and flux
# !SECTION

# SECTION Notebook 6 routines - Erasure Protocol?
# !SECTION



