# **Fokker-Planck: A Dynamic Integrator for Python**

### Installation

To install this package through pip use:

`pip install fokker-planck`

<br />

## Table of Contents

- [Overview](#overview)
- [The Fokker Planck Equation](#fokker-planck-equation)
- [A Preview](#a-preview)
- [1D Simulations](#1d-simulations)
- [2D Simulations](#2d-simulations)
- [The Simulator Interface](#the-simulator-interface)
- [In the Weeds: Integrator Options](#in-the-weeds-intgrator-details-and-options)
- [Additional Resources](#additional-resources)
- [References](#references)

## Overview

The intent of this package is to provide simple and user-friendly access to a Fokker-Planck equation integration suite that makes is relatively quick and simple for the end-user to run simulations of stockastic systems within a dynamic potential energy function, as well as trcking and investigating the steady state and time-dpendent properties of physical observables and quantities of interst, like work, power, and flux.

This is made possible by abstracting away many of the low-level detaisl of the integrator itself, and focusing on providing a simple interface that allows users to rapidly build pupose-built simulations, and allows for the flexibility of the system to be generalizable, extensible, and modular enough to be improved upon easily.

At its heart, this package is interested in integrating/solving the Fokker-Planck equation

In this `README` we first discuss the basics of the Fokker-Planck equation as it is often used, and then move onto the base case of a 1D integration, then briefly discuss the 2-dimensional extension. Following this, we introduce the Simulator interface, and show how its possible to run several nontriial simulations of physical system behaviour through its use, and the necessary requirements to customize that behaviour for specific purpose-driven investigations.

## Fokker Planck equation

Simply put, this software package deals with the Fokker-Planck equation. This equation represents the time-evolution of the probability distribution over states for a system that evolves under drift and diffusion alone. In fact, one can derive the FPE from the effective 'continuity' equation for stochastic processes, the Chapman Kolmogorov equation (see Chapter 3 of Ref.[2]). This means that the resulting behaviour is diffusive in nature, and will not capture heavy-tailed distributions or situations where anomalous diffusion takes place. For that, we would need to model a jump kernel explicitly into the equation, or support fractional derivative terms, which we do not.

However, in full generality the FPE representing this type of evolution can be represented in $N$-spatial dimensions as

$$ \partial_t p(\boldsymbol{x}, t) = \sum_i \partial_{{x_i}} \left[\mu(\boldsymbol{x}, t) p(x, t)\right] + \sum_{i, j}\partial_{x_i, x_j}^2D_{ij}p(\boldsymbol{x}, t) $$

which is rather complicated, but essentially allows for arbitrary mobilities $\mu$ as a function of the state vector $\boldsymbol{x}$

## A Preview

Before delving into more detail on the individual components and their uses, we breifly show in this section how to run a simple simulation of a diffusing particle in a 1D harmonic trapping potential, through the use of the `Simulator` interface.

The code snippet below shows how simple such a procedure is, starting with a system initialized from a uniform distribution an iterating the dynamics for 100 steps

```python
from fokker_planck.simulator import simulator

fpe_config = {
    "D": 1.0,
    "dx": 0.01,
    "dt": 0.001,
    "x_min": -3,
    "x_max": 3
}

trap_stringth = 4
total_time = 1.0

fpe_sim = simulator.HarmonicEquilibrationSimulator(fpe_config, trap_strength)

sim_result = fpe_sim.run_simulation(total_time)

```

The resuly type will bw `SimulationResult` (or some custom subclass of that type) and will contain all of the desired information on the simulation run. Also, this type can obviously be subclassed and expanded to contain any desired information. For instance, in slightly more complex scenario, we can define our own energy and force functions, to simulate the behaviour of whatever systemwe want to simulate.

## 1D Simulations

As of this point, 1D simulations are all that are supported by the integrator. This means that the current version of the integrator is restricted to model systems of the form:

## 2D Simulations

Not yet implemented: This is a feature that is coming down the pipline soon!

## The _Simulator_ Interface

## In the Weeds: Intgrator Details and Options

To understand how each of the components of the integrator work, there are a series of notebooks located in the `notebooks/functionality` directory of the source code. Specifically, because the implementation makes use of integrator splitting, we have separate logic for:

- Advection (`01-slarge-advection.ipynb`)
- Diffusion (`02-slarge-diffusion.ipynb`)
- Integrator Splitting (`03-slarge-operator-splitting.ipynb`)
- Advection-Diffusion (`04-slarge-advection-diffusion.ipynb`)
- Breathing Harmonic Trap (`05-slarge-breathing-trap.ipynb`)
- Periodic System (`06-slarge-periodic-system.ipynb`)

These notebooks go over the raw functionality, as well as go over a few of the model systems investigated in detail in the supporting documentation.

## Additional Resources

## References

<ol>
    <li>"Thermodynamic Metrics and Optimal Paths", D.A. Sivak & G.E. Crooks, <i>Phys. Rev. Lett.</i>, <b>2012</b></li>
    <li>"Optimal Control of Rotary Motors"</li>
    <li>"Stochastic Control in Microscopic Nonequilibrium Systems"</li>

</ol>
