"""Pixel integration — the camera's 1 px sampling aperture.

Single responsibility: the top-hat spanning -0.5..+0.5 px (each pixel
integrates the light landing on it). Always present in the detected
profile; the Thompson width chain deliberately EXCLUDES it (the bound
carries pixelation as its own a^2/12/N term — see width.py).
Same (x0, k) convention as gaussian_kernel.
"""
import numpy as np


def pixel_kernel(dx: float):
    n = max(int(round(1.0 / dx)), 1)
    return -0.5, np.ones(n + 1)
