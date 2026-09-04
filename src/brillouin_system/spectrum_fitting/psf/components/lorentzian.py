"""The Lorentzian line — the VIPA transmission profile.

Single responsibility: evaluate the unit-height Lorentzian
    L(x) = 1 / (1 + ((x - cen) / gamma)^2)
on a grid. gamma is the HWHM in pixels. This is the only component that
carries the physics (the fitted line); everything else in components/
describes the detection chain that smears it.
"""
import numpy as np


def lorentzian(x, cen, gamma):
    gamma = max(float(gamma), 1e-12)
    return 1.0 / (1.0 + ((np.asarray(x, dtype=float) - float(cen))
                         / gamma) ** 2)
