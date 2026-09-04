"""Monte Carlo frame generator with exactly-known noise.

Takes a noiseless mean frame [counts] and produces synthetic noisy frames
with the two-component camera noise model:

* shot noise — Poisson in ELECTRONS. Counts are converted with the camera
  gain g [e-/count], drawn, and converted back, so the count-domain variance
  is mean/g per pixel (this is why the gain is the input that "sets" the
  shot noise — there is no separate knob).
* read noise — additive Gaussian per pixel, rms in counts, independent of
  signal.

Because the truth and the noise are exactly known, fitting the generated
frames answers the question measured data cannot: how far above the photon
limit does a given estimator sit, with no instrument drift mixed in.

The model is deliberately minimal: no flicker (multiplicative illumination
noise is session-dependent and largely cancels in a row sum), no EMCCD
excess-noise factor, no dark current (measured negligible on our EMCCD at
operating settings). Add terms only when a measurement demands them.
"""
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np


@dataclass
class MonteCarloFrames:
    """Generate synthetic camera frames around a known truth.

    mean_frame        noiseless expectation of the frame [counts].
    gain_e_per_count  camera gain g [electrons/count]; sets the shot noise
                      (count-domain shot variance = mean/g). E.g. 3.89 for
                      the instrument in the paper, from photon-transfer
                      calibration.
    read_noise_counts per-pixel read noise rms [counts] (e.g. 1.10).
    n_images          number of frames a full run produces.
    seed              optional RNG seed for reproducible runs.
    """

    mean_frame: np.ndarray
    gain_e_per_count: float
    read_noise_counts: float
    n_images: int
    seed: int | None = None

    def __post_init__(self):
        self.mean_frame = np.asarray(self.mean_frame, dtype=float)
        if self.gain_e_per_count <= 0.0:
            raise ValueError("gain_e_per_count must be > 0.")
        if self.read_noise_counts < 0.0:
            raise ValueError("read_noise_counts must be >= 0.")
        if self.n_images < 1:
            raise ValueError("n_images must be >= 1.")

    def expected_std(self) -> np.ndarray:
        """Per-pixel std of the generated frames [counts] — the model itself.

        sqrt(mean/g + read^2); useful as the exact sigma for weighted fits
        and for verifying the generator.
        """
        lam = np.clip(self.mean_frame, 0.0, None)
        return np.sqrt(lam / self.gain_e_per_count + self.read_noise_counts ** 2)

    def frames(self) -> Iterator[np.ndarray]:
        """Yield n_images noisy frames [counts], one at a time.

        A generator, so a run over many large frames never holds the whole
        stack in memory. Use stack() when the full array is wanted.
        """
        rng = np.random.default_rng(self.seed)
        g = self.gain_e_per_count
        lam_e = np.clip(self.mean_frame, 0.0, None) * g
        for _ in range(self.n_images):
            shot_counts = rng.poisson(lam_e) / g
            read = rng.normal(0.0, self.read_noise_counts,
                              size=self.mean_frame.shape)
            yield shot_counts + read

    def stack(self) -> np.ndarray:
        """All frames as one (n_images, *frame_shape) array [counts]."""
        return np.stack(list(self.frames()))

    def run(self, fit: Callable[[np.ndarray], object]) -> list:
        """Apply any per-frame estimator and collect its results.

        `fit` takes one frame [counts] and returns anything (a float, a
        tuple, a FittedSpectrum, ...). Returns the n_images results in
        order. The scatter of these results is what the Thompson bound
        (noise_analysis.thompson) is compared against.
        """
        return [fit(frame) for frame in self.frames()]
