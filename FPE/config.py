# Configuration file for FPE integrators

DEFAULTS = {
    "diffScheme": "crank-nicholson",
    "adScheme": "lax-wendroff",
    "boundaryConditions": "hard-wall",
    "splittingMethod": "strang"
}

diffSchemes = [
    "explicit",
    "implicit",
    "crank-nicolson"
]

adSchemes = [
    "lax-wendroff"
]

boundaryConditions = [
    "open",
    "periodic",
    "hard-wall"
]

splittingMethods = [
    "lie",
    "strang",
    "swss"
]

