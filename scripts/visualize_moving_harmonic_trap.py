#!/usr/bin/env python3
"""Generate seaborn figures and animations of a Gaussian in a moving harmonic trap.

Examples
--------
    uv run python scripts/visualize_moving_harmonic_trap.py
    uv run python scripts/visualize_moving_harmonic_trap.py --dim 1d
    uv run python scripts/visualize_moving_harmonic_trap.py --dim 2d --no-animate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 1D/2D Fokker-Planck evolution in a moving harmonic trap."
    )
    parser.add_argument(
        "--dim", choices=("1d", "2d", "all"), default="all",
        help="Which integrator demonstration to render.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "figures",
        help="Directory for PNG/GIF/MP4 outputs.",
    )
    parser.add_argument(
        "--animate", action=argparse.BooleanOptionalAction, default=True,
        help="Write GIF animations (and MP4 if ffmpeg is available).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open interactive windows after saving (skips the Agg backend).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    from fokker_planck.viz import (
        animate_moving_trap_1d,
        animate_moving_trap_2d,
        plot_moving_trap_1d,
        plot_moving_trap_2d,
        simulate_moving_trap_1d,
        simulate_moving_trap_2d,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.dim in ("1d", "all"):
        print("Integrating 1D moving harmonic trap...")
        result_1d = simulate_moving_trap_1d()
        png_1d = args.outdir / "moving_trap_1d.png"
        written.append(plot_moving_trap_1d(result_1d, png_1d))
        print(f"  wrote {png_1d}")
        if args.animate:
            print("  rendering 1D animation...")
            anim_1d = animate_moving_trap_1d(
                result_1d, args.outdir / "moving_trap_1d.gif"
            )
            written.extend(anim_1d)
            for path in anim_1d:
                print(f"  wrote {path}")

    if args.dim in ("2d", "all"):
        print("Integrating 2D elliptical moving harmonic trap...")
        result_2d = simulate_moving_trap_2d()
        png_2d = args.outdir / "moving_trap_2d.png"
        written.append(plot_moving_trap_2d(result_2d, png_2d))
        print(f"  wrote {png_2d}")
        if args.animate:
            print("  rendering 2D animation...")
            anim_2d = animate_moving_trap_2d(
                result_2d, args.outdir / "moving_trap_2d.gif"
            )
            written.extend(anim_2d)
            for path in anim_2d:
                print(f"  wrote {path}")

    print("Done.")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
