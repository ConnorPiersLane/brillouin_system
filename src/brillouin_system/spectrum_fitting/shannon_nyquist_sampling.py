import numpy as np
from pathlib import Path
import hashlib


class ShannonNyquistSampling:
    def __init__(
        self,
        hwhm_min_px=0.25,
        hwhm_max_px=3.0,
        n_hwhm=41,
        n_phase=1001,
        oversample=1000,
        n_pixels=31,
        cache_round_decimals=6,
        auto_load_or_build=True,
    ):
        if n_pixels % 2 == 0:
            raise ValueError("n_pixels must be odd.")
        if hwhm_min_px <= 0:
            raise ValueError("hwhm_min_px must be positive.")
        if hwhm_max_px <= hwhm_min_px:
            raise ValueError("hwhm_max_px must be greater than hwhm_min_px.")

        self.hwhm_min_px = float(hwhm_min_px)
        self.hwhm_max_px = float(hwhm_max_px)
        self.n_hwhm = int(n_hwhm)
        self.n_phase = int(n_phase)
        self.oversample = int(oversample)
        self.n_pixels = int(n_pixels)
        self.cache_round_decimals = int(cache_round_decimals)

        self.hwhm_grid = np.linspace(self.hwhm_min_px, self.hwhm_max_px, self.n_hwhm)
        self.phase_true_grid = np.linspace(-0.5, 0.5, self.n_phase, endpoint=False)

        half = self.n_pixels // 2
        self.sample_pixels = np.arange(-half, half + 1, dtype=float)
        self.x_dense = np.linspace(-1.5, 1.5, int(3 * self.oversample) + 1)

        self._lut_cache = {}

        if auto_load_or_build:
            self.load_or_build_lut(verbose=True)

    # ----------------------------
    # Low-level model functions
    # ----------------------------

    @staticmethod
    def _lorentzian_pixel_integral(n, center, hwhm):
        left = n - 0.5
        right = n + 0.5
        g = float(hwhm)
        return g * (
            np.arctan((right - center) / g) - np.arctan((left - center) / g)
        )

    @staticmethod
    def _sinc_reconstruct(x, n, samples):
        return np.sum(samples[:, None] * np.sinc(x[None, :] - n[:, None]), axis=0)

    @staticmethod
    def _estimate_peak_argmax(x, y):
        return float(x[np.argmax(y)])

    @staticmethod
    def _deduplicate_sorted(x, y, tol=1e-14):
        order = np.argsort(x)
        x = np.asarray(x)[order]
        y = np.asarray(y)[order]
        keep = np.ones(len(x), dtype=bool)
        if len(x) > 1:
            keep[1:] = np.diff(x) > tol
        return x[keep], y[keep]

    @staticmethod
    def _wrap_phase_signed(phase):
        return ((phase + 0.5) % 1.0) - 0.5

    # ----------------------------
    # LUT metadata / paths
    # ----------------------------

    def _metadata(self):
        return {
            "hwhm_min_px": self.hwhm_min_px,
            "hwhm_max_px": self.hwhm_max_px,
            "n_hwhm": self.n_hwhm,
            "n_phase": self.n_phase,
            "oversample": self.oversample,
            "n_pixels": self.n_pixels,
            "cache_round_decimals": self.cache_round_decimals,
        }

    def _metadata_signature(self):
        items = tuple(sorted(self._metadata().items()))
        return hashlib.md5(repr(items).encode("utf-8")).hexdigest()[:12]

    def _script_directory(self):
        try:
            return Path(__file__).resolve().parent
        except NameError:
            return Path.cwd()

    def _default_lut_path(self):
        sig = self._metadata_signature()
        return self._script_directory() / f"shannon_nyquist_lut_{sig}.npz"

    # ----------------------------
    # LUT construction
    # ----------------------------

    def _cache_key(self, hwhm_px):
        return round(float(hwhm_px), self.cache_round_decimals)

    def _build_single_lut(self, hwhm_px):
        key = self._cache_key(hwhm_px)
        if key in self._lut_cache:
            return self._lut_cache[key]

        phase_true = self.phase_true_grid.copy()
        phase_rec = np.empty_like(phase_true)

        n = self.sample_pixels
        x = self.x_dense

        for i, d in enumerate(phase_true):
            s = self._lorentzian_pixel_integral(n=n, center=d, hwhm=hwhm_px)
            y = self._sinc_reconstruct(x=x, n=n, samples=s)
            xpk = self._estimate_peak_argmax(x, y)
            phase_rec[i] = self._wrap_phase_signed(xpk)

        rec_sorted, true_sorted = self._deduplicate_sorted(phase_rec, phase_true)

        lut = {
            "hwhm_px": float(hwhm_px),
            "phase_true": phase_true,
            "phase_rec": phase_rec,
            "phase_rec_sorted": rec_sorted,
            "phase_true_sorted_by_rec": true_sorted,
        }
        self._lut_cache[key] = lut
        return lut

    def build_cache(self, verbose=False):
        for i, h in enumerate(self.hwhm_grid):
            self._build_single_lut(float(h))
            if verbose and ((i + 1) % max(1, self.n_hwhm // 10) == 0 or i == self.n_hwhm - 1):
                print(f"Built {i+1}/{self.n_hwhm} LUTs")

    # ----------------------------
    # Save / load
    # ----------------------------

    def save_lut(self, filename=None):
        if filename is None:
            filename = self._default_lut_path()
        filename = Path(filename)

        phase_rec_table = np.full((self.n_hwhm, self.n_phase), np.nan, dtype=float)
        built_mask = np.zeros(self.n_hwhm, dtype=bool)

        for i, h in enumerate(self.hwhm_grid):
            key = self._cache_key(h)
            if key in self._lut_cache:
                built_mask[i] = True
                phase_rec_table[i, :] = self._lut_cache[key]["phase_rec"]

        np.savez_compressed(
            filename,
            metadata=np.array([self._metadata()], dtype=object),
            hwhm_grid=self.hwhm_grid,
            phase_true_grid=self.phase_true_grid,
            phase_rec_table=phase_rec_table,
            built_mask=built_mask,
        )

    def load_lut(self, filename=None):
        if filename is None:
            filename = self._default_lut_path()
        filename = Path(filename)

        data = np.load(filename, allow_pickle=True)
        metadata = data["metadata"][0]

        for key, expected in self._metadata().items():
            found = metadata[key]
            if found != expected:
                raise ValueError(
                    f"LUT mismatch for '{key}': file={found}, current={expected}"
                )

        if not np.allclose(data["hwhm_grid"], self.hwhm_grid):
            raise ValueError("LUT mismatch: hwhm_grid does not match current settings.")
        if not np.allclose(data["phase_true_grid"], self.phase_true_grid):
            raise ValueError("LUT mismatch: phase_true_grid does not match current settings.")

        phase_rec_table = data["phase_rec_table"]
        built_mask = data["built_mask"]

        self._lut_cache.clear()

        for i, h in enumerate(self.hwhm_grid):
            if not built_mask[i]:
                continue

            phase_rec = phase_rec_table[i]
            rec_sorted, true_sorted = self._deduplicate_sorted(
                phase_rec, self.phase_true_grid
            )

            self._lut_cache[self._cache_key(h)] = {
                "hwhm_px": float(h),
                "phase_true": self.phase_true_grid.copy(),
                "phase_rec": phase_rec.copy(),
                "phase_rec_sorted": rec_sorted,
                "phase_true_sorted_by_rec": true_sorted,
            }

    def load_or_build_lut(self, verbose=False):
        filename = self._default_lut_path()

        if filename.exists():
            if verbose:
                print(f"Loading LUT from: {filename}")
            self.load_lut(filename)
        else:
            if verbose:
                print(f"No LUT found at: {filename}")
                print("Building LUT...")
            self.build_cache(verbose=verbose)
            self.save_lut(filename)
            if verbose:
                print(f"Saved LUT to: {filename}")

    # ----------------------------
    # Phase inversion
    # ----------------------------

    def _invert_phase_for_hwhm(self, rec_phase, hwhm_px):
        lut = self._build_single_lut(hwhm_px)
        x = lut["phase_rec_sorted"]
        y = lut["phase_true_sorted_by_rec"]

        rec_phase = np.asarray(rec_phase, dtype=float)
        rec_phase = self._wrap_phase_signed(rec_phase)
        rec_phase_clip = np.clip(rec_phase, x[0], x[-1])

        true_phase = np.interp(rec_phase_clip, x, y)
        return self._wrap_phase_signed(true_phase)

    def _invert_phase_interpolated_hwhm(self, rec_phase, hwhm_px):
        h = float(hwhm_px)

        if h <= self.hwhm_grid[0]:
            return self._invert_phase_for_hwhm(rec_phase, self.hwhm_grid[0])
        if h >= self.hwhm_grid[-1]:
            return self._invert_phase_for_hwhm(rec_phase, self.hwhm_grid[-1])

        j = np.searchsorted(self.hwhm_grid, h)
        h0 = self.hwhm_grid[j - 1]
        h1 = self.hwhm_grid[j]
        t = (h - h0) / (h1 - h0)

        p0 = self._invert_phase_for_hwhm(rec_phase, h0)
        p1 = self._invert_phase_for_hwhm(rec_phase, h1)

        dp = p1 - p0
        dp = ((dp + 0.5) % 1.0) - 0.5

        return self._wrap_phase_signed(p0 + t * dp)

    # ----------------------------
    # Public API
    # ----------------------------

    def correct_center(self, rec_center_px, hwhm_px):
        rc = np.asarray(rec_center_px, dtype=float)
        k = np.floor(rc + 0.5)
        rec_phase = rc - k
        true_phase = self._invert_phase_interpolated_hwhm(rec_phase, hwhm_px)
        true_center = k + true_phase
        if true_center.ndim == 0:
            return float(true_center)
        return true_center

if __name__ == "__main__":
    sns = ShannonNyquistSampling(
        hwhm_min_px=0.25,
        hwhm_max_px=3.0,
        n_hwhm=41,
        n_phase=1001,
        oversample=1000,
        n_pixels=11,
        auto_load_or_build=True,
    )

    rec_center_px = 12.37
    hwhm_px = 2.0
    true_center_px = sns.correct_center(rec_center_px, hwhm_px)

    print("corrected center:", true_center_px)