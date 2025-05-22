import numpy as np
from timeit import default_timer as timer

from scipy.optimize import curve_fit


def _linear_model(x, sd, fsr):
    return 0.5 * fsr - 0.5 * sd * x

def fit_calibration_curve(px_dist, freq, xtol=1e-6, ftol=1e-6, maxfev=500) -> tuple[float, float]:

    px_dist = np.asarray(px_dist, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)

    start_time = timer()
    initial_guess = [0.127, 21.5]

    try:
        popt, _ = curve_fit(
            _linear_model,
            px_dist,
            freq,
            p0=initial_guess,
            xtol=xtol,
            ftol=ftol,
            maxfev=maxfev
        )

        fitted_values = _linear_model(px_dist, *popt)
        residuals = freq - fitted_values
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((freq - np.mean(freq)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f'[fit_calibration_curve] Calibration curve fitting time = {(timer() - start_time) * 1e3:.2f} ms')
        print(f'[fit_calibration_curve] R^2 = {r_squared:.4f}')

        sd, fsr = popt

    except Exception as e:
        print(f"[DataFitting] Calibration fitting failed: {e}")
        sd = np.nan
        fsr = np.nan

    return sd, fsr