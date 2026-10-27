"""Seaborn-styled figures and animations for moving harmonic traps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from fokker_planck.integrator import FokkerPlanck1D
from fokker_planck.integrator_2D import FokkerPlanck2D
from fokker_planck.types.basetypes import BoundaryCondition
import fokker_planck.forceFunctions as ff

PALETTE = {
    "paper": "#f6f1ea",
    "panel": "#fffaf4",
    "ink": "#241f1c",
    "muted": "#7a716a",
    "trap": "#2f6f9c",
    "mean": "#c44536",
    "theory": "#241f1c",
    "fill": "#d8ecec",
    "grid": "#b7b7b4",
}

DENSITY_BG = "#e2e2e0"
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "prob_reds",
    [
        DENSITY_BG,
        "#ead7d3",
        "#e0b4ad",
        "#d49288",
        "#c8786e",
        "#c06a62",
    ],
)


def apply_style() -> None:
    sns.set_theme(
        context="talk",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
            "grid.color": "#eadfd4",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.facecolor": PALETTE["paper"],
            "savefig.bbox": "tight",
        },
    )


def lagging_mean(time: np.ndarray, x0: float, velocity: float, D: float, k: float) -> np.ndarray:
    """Mean of an overdamped Gaussian chasing a trap that moves at constant speed."""
    return x0 + velocity * time - (velocity / (D * k)) * (1.0 - np.exp(-D * k * time))


@dataclass
class MovingTrap1D:
    x: np.ndarray
    time: np.ndarray
    density: np.ndarray
    trap: np.ndarray
    mean: np.ndarray
    k: float
    D: float
    velocity: float


@dataclass
class MovingTrap2D:
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    density: np.ndarray
    trap: np.ndarray
    mean: np.ndarray
    k: float
    D: float
    velocity: np.ndarray | None = None
    ellipse_center: np.ndarray | None = None
    ellipse_axes: np.ndarray | None = None
    start: np.ndarray | None = None
    t_approach: float = 0.0
    n_revolutions: float = 3.0


def simulate_moving_trap_1d(
    *,
    D: float = 1.0,
    k: float = 8.0,
    velocity: float = 1.35,
    x0: float = -1.05,
    tau: float = 1.15,
    dx: float = 0.04,
    dt: float = 0.0004,
    x_min: float = -2.5,
    x_max: float = 2.3,
    n_frames: int = 48,
) -> MovingTrap1D:
    x = np.arange(x_min, x_max, dx)
    fpe = FokkerPlanck1D(
        D, dt, dx, x, boundary_cond=BoundaryCondition.HardWall
    )
    fpe.initialize_probability(x0, 1.0 / k)

    n_steps = int(tau / dt)
    stride = max(1, n_steps // n_frames)
    times, dens, trap, mean = [], [], [], []

    for i in range(n_steps):
        t = i * dt
        center = x0 + velocity * t
        if i % stride == 0:
            times.append(t)
            dens.append(fpe.prob.copy())
            trap.append(center)
            mean.append(fpe.mean)
        fpe.integrate_step([k, center], ff.harmonic_force)

    return MovingTrap1D(
        x=x,
        time=np.asarray(times),
        density=np.asarray(dens),
        trap=np.asarray(trap),
        mean=np.asarray(mean),
        k=k,
        D=D,
        velocity=velocity,
    )


def _ellipse_xy(
    center: np.ndarray, axes: np.ndarray, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return center[0] + axes[0] * np.cos(theta), center[1] + axes[1] * np.sin(theta)


def _trap_center_ellipse(
    t: float,
    start: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    t_approach: float,
    omega: float,
    theta0: float = 0.0,
) -> np.ndarray:
    """Off-ellipse start, then constant-speed motion around the ellipse."""
    join = np.array([
        center[0] + axes[0] * np.cos(theta0),
        center[1] + axes[1] * np.sin(theta0),
    ])
    if t_approach <= 0.0 or t >= t_approach:
        theta = theta0 + omega * max(t - t_approach, 0.0)
        return np.array([
            center[0] + axes[0] * np.cos(theta),
            center[1] + axes[1] * np.sin(theta),
        ])
    s = np.clip(t / t_approach, 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    return start + s * (join - start)


def simulate_moving_trap_2d(
    *,
    D: float = 1.0,
    k: float = 6.0,
    start: tuple[float, float] = (0.0, 0.0),
    ellipse_center: tuple[float, float] = (0.0, 0.0),
    ellipse_axes: tuple[float, float] = (1.18, 0.72),
    t_approach: float = 0.45,
    period: float = 1.05,
    n_revolutions: float = 3.0,
    theta0: float = 0.0,
    dx: float = 0.08,
    dt: float = 0.001,
    span: float = 2.35,
    n_frames: int = 150,
) -> MovingTrap2D:
    start_xy = np.asarray(start, dtype=float)
    center = np.asarray(ellipse_center, dtype=float)
    axes = np.asarray(ellipse_axes, dtype=float)
    omega = 2.0 * np.pi / period
    tau = t_approach + n_revolutions * period

    x = np.arange(-span, span, dx)
    y = np.arange(-span, span, dx)
    fpe = FokkerPlanck2D(
        D, dt, dx, dx, x, y, boundary_cond=BoundaryCondition.HardWall
    )
    fpe.initialize_probability(start_xy, (1.0 / k, 1.0 / k))

    n_steps = int(round(tau / dt))
    stride = max(1, n_steps // n_frames)
    times, dens, trap, mean = [], [], [], []

    for i in range(n_steps):
        t = i * dt
        cx, cy = _trap_center_ellipse(
            t, start_xy, center, axes, t_approach, omega, theta0
        )
        params = ([k, cx], [k, cy])
        if i % stride == 0 or i == n_steps - 1:
            times.append(t)
            dens.append(fpe.prob_grid.copy())
            trap.append((cx, cy))
            mean.append(fpe.mean.copy())
        fpe.integrate_step(params, ff.harmonic_force_2D_x, ff.harmonic_force_2D_y)

    return MovingTrap2D(
        x=x,
        y=y,
        time=np.asarray(times),
        density=np.asarray(dens),
        trap=np.asarray(trap),
        mean=np.asarray(mean),
        k=k,
        D=D,
        ellipse_center=center,
        ellipse_axes=axes,
        start=start_xy,
        t_approach=t_approach,
        n_revolutions=n_revolutions,
    )


def plot_moving_trap_1d(result: MovingTrap1D, out_path: Path) -> Path:
    apply_style()
    n = len(result.time)
    colors = sns.color_palette("mako", n_colors=n)
    theory = lagging_mean(result.time, result.trap[0], result.velocity, result.D, result.k)

    fig = plt.figure(figsize=(11.6, 7.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.32)

    ax_p = fig.add_subplot(gs[0, :])
    for i, (p, color) in enumerate(zip(result.density, colors)):
        alpha = 0.18 if i not in (0, n - 1) else 0.42
        lw = 1.1 if i not in (0, n - 1) else 2.2
        ax_p.plot(result.x, p, color=color, lw=lw, alpha=min(0.95, 0.35 + 0.65 * i / n))
        if i in (0, n - 1):
            ax_p.fill_between(result.x, p, color=color, alpha=alpha, linewidth=0)
    ax_p.axvline(result.trap[0], color=PALETTE["trap"], ls="--", lw=1.2, alpha=0.7)
    ax_p.axvline(result.trap[-1], color=PALETTE["trap"], ls="-", lw=1.6, alpha=0.9)
    ax_p.set_xlabel("position $x$")
    ax_p.set_ylabel("probability density $p(x,t)$")
    ax_p.set_title("Gaussian density chasing a moving harmonic trap (1D)")
    sm = plt.cm.ScalarMappable(
        cmap=sns.color_palette("mako", as_cmap=True),
        norm=Normalize(result.time.min(), result.time.max()),
    )
    cbar = fig.colorbar(sm, ax=ax_p, pad=0.015, fraction=0.03)
    cbar.set_label("time $t$")
    sns.despine(ax=ax_p)

    ax_kymo = fig.add_subplot(gs[1, 0])
    T, X = np.meshgrid(result.time, result.x, indexing="ij")
    mesh = ax_kymo.pcolormesh(
        X, T, result.density, cmap="mako", shading="auto", rasterized=True
    )
    ax_kymo.plot(result.trap, result.time, color=PALETTE["trap"], lw=2.0, label="trap")
    ax_kymo.plot(result.mean, result.time, color="white", lw=1.8, label="mean")
    ax_kymo.set_xlabel("position $x$")
    ax_kymo.set_ylabel("time $t$")
    ax_kymo.set_title("Space–time density")
    ax_kymo.legend(loc="lower right", fontsize=11)
    fig.colorbar(mesh, ax=ax_kymo, pad=0.02, fraction=0.046, label="$p(x,t)$")
    sns.despine(ax=ax_kymo)

    ax_mean = fig.add_subplot(gs[1, 1])
    ax_mean.plot(result.time, result.trap, color=PALETTE["trap"], lw=2.3, label="trap center")
    ax_mean.plot(result.time, result.mean, color=PALETTE["mean"], lw=2.3, label=r"sim. $\langle x\rangle$")
    ax_mean.plot(
        result.time, theory, color=PALETTE["theory"], ls="--", lw=1.6, label="analytic lag"
    )
    ax_mean.set_xlabel("time $t$")
    ax_mean.set_ylabel("position")
    ax_mean.set_title("Mean lags the moving well")
    ax_mean.legend(loc="upper left", fontsize=11)
    sns.despine(ax=ax_mean)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _density_extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    dx = 0.5 * (x[1] - x[0])
    dy = 0.5 * (y[1] - y[0])
    return [x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy]


def _add_xy_grid(ax, extent: list[float], spacing: float = 0.5, label_spacing: float = 1.0) -> None:
    x0, x1, y0, y1 = extent
    xticks = np.arange(np.ceil(x0 / spacing) * spacing, x1 + 1e-9, spacing)
    yticks = np.arange(np.ceil(y0 / spacing) * spacing, y1 + 1e-9, spacing)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([
        f"{tick:g}" if abs(tick / label_spacing - round(tick / label_spacing)) < 1e-8 else ""
        for tick in xticks
    ])
    ax.set_yticklabels([
        f"{tick:g}" if abs(tick / label_spacing - round(tick / label_spacing)) < 1e-8 else ""
        for tick in yticks
    ])
    ax.set_facecolor(DENSITY_BG)
    for x in xticks:
        ax.axvline(x, color=PALETTE["grid"], lw=0.55, alpha=0.38, zorder=1)
    for y in yticks:
        ax.axhline(y, color=PALETTE["grid"], lw=0.55, alpha=0.38, zorder=1)


def plot_moving_trap_2d(result: MovingTrap2D, out_path: Path) -> Path:
    apply_style()
    frames = np.linspace(0, len(result.time) - 1, 6, dtype=int)
    vmax = np.percentile(result.density, 99.5)
    extent = _density_extent(result.x, result.y)
    ell_x, ell_y = _ellipse_xy(result.ellipse_center, result.ellipse_axes)

    fig = plt.figure(figsize=(12.8, 9.6), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08, w_pad=0.04, hspace=0.06, wspace=0.04)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0])
    gs_top = gs[0].subgridspec(2, 3, wspace=0.08, hspace=0.12)
    gs_bot = gs[1].subgridspec(1, 2, wspace=0.18)

    snapshot_axes = []
    image = None
    for panel, idx in enumerate(frames):
        ax = fig.add_subplot(gs_top[panel // 3, panel % 3])
        snapshot_axes.append(ax)
        image = ax.imshow(
            result.density[idx].T,
            origin="lower",
            extent=extent,
            cmap=DENSITY_CMAP,
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
            zorder=0,
        )
        ax.plot(ell_x, ell_y, ls="--", color=PALETTE["trap"], lw=1.15, alpha=0.55, zorder=2)
        cx, cy = result.trap[idx]
        mx, my = result.mean[idx]
        ax.plot([mx, cx], [my, cy], color=PALETTE["ink"], lw=1.1, alpha=0.45, zorder=3)
        ax.scatter(
            [cx], [cy], s=42, c=PALETTE["trap"], edgecolors="white",
            linewidths=0.8, zorder=4,
        )
        ax.scatter(
            [mx], [my], s=28, c=PALETTE["mean"], edgecolors="white",
            linewidths=1.1, zorder=5,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        _add_xy_grid(ax, extent)
        ax.set_title(f"$t = {result.time[idx]:.2f}$", fontsize=13)
        if panel % 3 == 0:
            ax.set_ylabel("$y$")
        else:
            ax.set_yticklabels([])
        if panel // 3 == 1:
            ax.set_xlabel("$x$")
        else:
            ax.set_xticklabels([])
        sns.despine(ax=ax)

    cbar = fig.colorbar(image, ax=snapshot_axes, fraction=0.025, pad=0.02)
    cbar.set_label("$p(x,y)$")

    ax_tr = fig.add_subplot(gs_bot[0, 0])
    ax_tr.plot(
        ell_x, ell_y, ls="--", color=PALETTE["trap"], lw=1.8, alpha=0.85,
        label="potential minima",
    )
    ax_tr.plot(
        result.trap[:, 0], result.trap[:, 1], color=PALETTE["trap"],
        lw=2.2, alpha=0.35, label="trap path",
    )
    ax_tr.plot(
        result.mean[:, 0], result.mean[:, 1], color=PALETTE["mean"],
        lw=2.2, label="density mean",
    )
    ax_tr.scatter(
        result.start[0], result.start[1], c=PALETTE["ink"], s=55,
        marker="x", zorder=5, label="start",
    )
    ax_tr.set_aspect("equal")
    ax_tr.set_xlabel("$x$")
    ax_tr.set_ylabel("$y$")
    ax_tr.set_title(f"Off-ellipse start, then {result.n_revolutions:g} revolutions")
    ax_tr.legend(loc="upper right", fontsize=10)
    sns.despine(ax=ax_tr)

    ax_xy = fig.add_subplot(gs_bot[0, 1])
    ax_xy.axvline(
        result.t_approach, color=PALETTE["muted"], ls=":", lw=1.4, label="joins ellipse"
    )
    ax_xy.plot(result.time, result.trap[:, 0], color=PALETTE["trap"], lw=2.2, label=r"trap $x$")
    ax_xy.plot(result.time, result.trap[:, 1], color=PALETTE["trap"], lw=1.6, ls=":", alpha=0.9, label=r"trap $y$")
    ax_xy.plot(result.time, result.mean[:, 0], color=PALETTE["mean"], lw=2.2, label=r"$\langle x\rangle$")
    ax_xy.plot(result.time, result.mean[:, 1], color=PALETTE["mean"], lw=1.6, ls=":", alpha=0.9, label=r"$\langle y\rangle$")
    ax_xy.set_xlabel("time $t$")
    ax_xy.set_ylabel("position")
    ax_xy.set_title(f"{result.n_revolutions:g} revolutions after the approach")
    ax_xy.legend(loc="lower left", fontsize=10, ncol=2)
    sns.despine(ax=ax_xy)

    fig.suptitle("Gaussian density in a 2D harmonic trap on an elliptical path", fontsize=16)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path


def _save_animation(anim: animation.FuncAnimation, out_path: Path, fps: int = 20) -> list[Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path = out_path.with_suffix(".gif")
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=140)
    written = [gif_path]

    if animation.writers.is_available("ffmpeg"):
        mp4_path = out_path.with_suffix(".mp4")
        anim.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=140)
        written.append(mp4_path)
    return written


def animate_moving_trap_1d(result: MovingTrap1D, out_path: Path) -> list[Path]:
    apply_style()
    theory = lagging_mean(result.time, result.trap[0], result.velocity, result.D, result.k)
    pot_scale = 0.22 * result.density.max() / (0.5 * result.k * (0.8 ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax, ax_m = axes
    ax.set_xlim(result.x.min(), result.x.max())
    ax.set_ylim(0.0, 1.18 * result.density.max())
    ax.set_xlabel("position $x$")
    ax.set_ylabel("$p(x,t)$")
    ax.set_title("Density and moving well")

    (pot_line,) = ax.plot([], [], color="#cbbba8", lw=2.0, alpha=0.9, label="trap $U(x)$")
    fill = ax.fill_between(result.x, result.density[0], color=PALETTE["mean"], alpha=0.28)
    (dens_line,) = ax.plot(result.x, result.density[0], color=PALETTE["mean"], lw=2.4)
    trap_line = ax.axvline(result.trap[0], color=PALETTE["trap"], lw=1.8)
    ax.legend(loc="upper right", fontsize=11)
    sns.despine(ax=ax)

    ax_m.set_xlim(result.time.min(), result.time.max())
    ypad = 0.15
    ax_m.set_ylim(min(result.trap.min(), result.mean.min()) - ypad,
                  max(result.trap.max(), result.mean.max()) + ypad)
    ax_m.plot(result.time, result.trap, color=PALETTE["trap"], lw=1.7, alpha=0.45)
    ax_m.plot(result.time, theory, color=PALETTE["ink"], ls="--", lw=1.3, alpha=0.7)
    (mean_line,) = ax_m.plot([], [], color=PALETTE["mean"], lw=2.4)
    (now_pt,) = ax_m.plot([], [], "o", color=PALETTE["trap"], ms=8)
    ax_m.set_xlabel("time $t$")
    ax_m.set_ylabel("position")
    ax_m.set_title("Lag behind the trap")
    ax_m.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["trap"], lw=1.7, label="trap"),
            Line2D([0], [0], color=PALETTE["mean"], lw=2.4, label=r"$\langle x\rangle$"),
            Line2D([0], [0], color=PALETTE["ink"], ls="--", lw=1.3, label="theory"),
        ],
        loc="upper left",
        fontsize=11,
    )
    sns.despine(ax=ax_m)
    fig.tight_layout()

    def update(i: int):
        nonlocal fill
        fill.remove()
        fill = ax.fill_between(
            result.x, result.density[i], color=PALETTE["mean"], alpha=0.28
        )
        dens_line.set_ydata(result.density[i])
        trap_line.set_xdata([result.trap[i], result.trap[i]])
        well = 0.5 * result.k * (result.x - result.trap[i]) ** 2
        pot_line.set_data(result.x, well * pot_scale)
        mean_line.set_data(result.time[: i + 1], result.mean[: i + 1])
        now_pt.set_data([result.time[i]], [result.trap[i]])
        return dens_line, trap_line, pot_line, mean_line, now_pt, fill

    anim = animation.FuncAnimation(
        fig, update, frames=len(result.time), interval=80, blit=False
    )
    saved = _save_animation(anim, out_path, fps=16)
    plt.close(fig)
    return saved


def animate_moving_trap_2d(result: MovingTrap2D, out_path: Path) -> list[Path]:
    apply_style()
    vmax = np.percentile(result.density, 99.6)
    extent = _density_extent(result.x, result.y)
    ell_x, ell_y = _ellipse_xy(result.ellipse_center, result.ellipse_axes)

    fig, ax = plt.subplots(figsize=(7.0, 6.8))
    image = ax.imshow(
        result.density[0].T,
        origin="lower",
        extent=extent,
        cmap=DENSITY_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
    )
    ax.plot(
        ell_x, ell_y, ls="--", color=PALETTE["trap"], lw=1.6, alpha=0.55, zorder=3,
        label="potential minima",
    )
    ax.scatter(
        [result.start[0]], [result.start[1]],
        marker="x", s=55, c=PALETTE["ink"], linewidths=1.4, zorder=4, label="start",
    )
    (trail,) = ax.plot(
        [], [], color=PALETTE["trap"], lw=2.2, alpha=0.28, zorder=4,
        label="trap path",
    )
    (mean_trail,) = ax.plot([], [], color=PALETTE["mean"], lw=1.6, alpha=0.85, zorder=4)
    (lag_line,) = ax.plot([], [], color=PALETTE["ink"], lw=1.2, alpha=0.45, zorder=5)
    trap_pt = ax.scatter(
        [result.trap[0, 0]], [result.trap[0, 1]],
        s=70, c=PALETTE["trap"], edgecolors="white", linewidths=1.0, zorder=6,
        label="trap",
    )
    mean_pt = ax.scatter(
        [result.mean[0, 0]], [result.mean[0, 1]],
        s=40, c=PALETTE["mean"], edgecolors="white", linewidths=1.2, zorder=7,
        label="mean",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _add_xy_grid(ax, extent)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="$p(x,y)$")
    sns.despine(ax=ax)
    fig.tight_layout()

    def update(i: int):
        image.set_data(result.density[i].T)
        trap_pt.set_offsets([result.trap[i]])
        mean_pt.set_offsets([result.mean[i]])
        trail.set_data(result.trap[: i + 1, 0], result.trap[: i + 1, 1])
        mean_trail.set_data(result.mean[: i + 1, 0], result.mean[: i + 1, 1])
        lag_line.set_data(
            [result.mean[i, 0], result.trap[i, 0]],
            [result.mean[i, 1], result.trap[i, 1]],
        )
        phase = "approach" if result.time[i] < result.t_approach else "orbit"
        title.set_text(
            f"Elliptical trap  ·  {phase}  ·  $t = {result.time[i]:.2f}$"
        )
        return image, trap_pt, mean_pt, trail, mean_trail, lag_line, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(result.time), interval=80, blit=False
    )
    saved = _save_animation(anim, out_path, fps=16)
    plt.close(fig)
    return saved
