# NA-integrated Brillouin lineshape — model & implementation

This documents the `na_lorentzian` / `na_gauss_lorentzian` fitting models
(and their `_window` variants) selectable in `SpectrumFitter`. It is the
ballistic (single-scattering) term of the aperture-integrated lineshape from
Mattarelli et al., *ACS Photonics* **9**, 2087 (2022), SI eq. 2s, specialized
to clear samples.

Code (paths relative to repo root):
- [`na_lineshape.py`](../src/brillouin_system/spectrum_fitting/na_lineshape.py) — the lineshape and its discretization
- [`spectrum_fitter.py`](../src/brillouin_system/spectrum_fitting/spectrum_fitter.py) — model selection, parameter setup, bounds
- [`na_correction5.py`](../src/brillouin_system/spectrum_fitting/na_correction5.py) — the angle-geometry helpers
- [`calibration.py`](../src/brillouin_system/calibration/calibration.py) — `elastic_anchors()`

---

## 1. Physics

A high-NA objective collects a **cone** of scattering directions around exact
backscattering (180°). A ray collected at deviation angle `v` from exact
backscattering (so its scattering angle is `θ = π − v`) exchanges wavevector

$$
q(v) = 2\,n\,k_0 \cos(v/2),
$$

where `n` is the sample refractive index and `k_0` the vacuum wavenumber. Since
the Brillouin frequency is proportional to `q` (acoustic dispersion
`ω_B = V q`), each collected ray resonates at

$$
f(v) = f_{180}\,\cos(v/2) \le f_{180}.
$$

So the recorded peak is a **superposition of sub-peaks**, every one at or below
`f_180`, weighted by how much of the collected light comes from angle `v`. This
biases a symmetric fit **low** and broadens/skews the peak. Fitting the
superposition directly returns `f_180` — the true 180° shift — as a parameter.

### Collection weight `W(v)`

Two selectable weights, both cut off at the aperture half-angle `α`:

**Uniform pupil** (`na_lorentzian`, the paper's model) — weight is pure solid
angle:

$$
W(v) = \sin v, \qquad 0 \le v \le \alpha .
$$

This is exactly the paper's `q\,dq` weight (`q ∝ cos(v/2)`, `dq ∝ sin(v/2)dv`,
and `sin v = 2\sin(v/2)\cos(v/2)`). Single geometric input `α`.

**Gaussian fiber-coupling** (`na_gauss_lorentzian`, the complete model) — the
solid angle apodized by the collection-fiber mode overlap:

$$
W(v) = \sin v \; \exp\!\left[-2\left(\frac{v}{v_0}\right)^{2}\right],
\qquad 0 \le v \le \alpha .
$$

`v_0` is the Gaussian coupling half-width (an angle). This is the
`na_correction5` weight validated on water.

> Only the **collection** cone is modeled; illumination is treated as an axial
> pencil ray (good for an underfilled objective). Multiple scattering is
> dropped (clear samples).

### Sub-peak core

Each sub-peak is a **Lorentzian** (near-resonance limit of the DHO; we only
want position). The NA broadening is carried entirely by the kernel, so the
fitted `γ` stays the *intrinsic* HWHM. The continuous lineshape is

$$
S(f) = A \,\frac{\displaystyle\int_0^{\alpha} W(v)\,
        L\!\big(f;\, f_{180}\cos(v/2),\, \gamma\big)\,dv}
       {\displaystyle\int_0^{\alpha} W(v)\,dv} \; + \; \text{offset},
\qquad
L(f; f_c, \gamma) = \frac{\gamma^2}{(f-f_c)^2 + \gamma^2}.
$$

---

## 2. Working in pixel space, anchored at the elastic line

The fit runs on the raw camera axis (pixels), not GHz. Because the NA downshift
is multiplicative on the frequency **measured from the elastic (Rayleigh)
line**, it maps cleanly to pixels: if the elastic order sits at pixel `R` and
the true 180° peak at pixel `c`, then the sub-peak for angle `v` sits at

$$
x_\text{sub}(v) = c - (c - R)\,\underbrace{\big(1 - \cos(v/2)\big)}_{\text{frac}(v)} .
$$

At `v = 0` (exact backscatter) the sub-peak is at `c`; as `v → α` it slides
toward `R`. `R` is a **fixed input** from the calibration (§4); `c` is the
fitted 180° position.

A VIPA spectrum has **two** Brillouin peaks (Stokes of order *n*,
anti-Stokes of order *n+1*), so each peak is anchored at its **own** elastic
order `R_left`, `R_right`.

---

## 3. Discretization (exactly what the code computes)

From `na_angular_grid(alpha, n_quad, v0)`:

- Quadrature nodes: `v_k = linspace(0, α, n_quad)`, default `n_quad = 41`.
- Weights: `W_k = sin(v_k)`, times `exp(-2 (v_k/v0)^2)` if `v0` is set.
- Downshift fractions: `frac_k = 1 − cos(v_k/2)`.
- Normalization: `wsum = ∫ W dv` via `np.trapezoid(W, v)` (raises if ≤ 0).

For each peak, with `x` the pixel axis and `(A, c, γ, R)` its parameters, the
sub-peak centers are `x_sub,k = c − (c − R)·frac_k`. Cores are **pixel-area
integrated** (the signal is a photon count per pixel), i.e. the Lorentzian is
integrated over each pixel's `[x−½, x+½]` — the same closed form as the plain
`_2lorentzian_binned` model, so amplitudes and widths are directly comparable:

$$
P_k(x) = A\,\gamma\left[\arctan\!\frac{x+\tfrac12 - x_{\text{sub},k}}{\gamma}
                      - \arctan\!\frac{x-\tfrac12 - x_{\text{sub},k}}{\gamma}\right].
$$

The per-peak model is the weighted average over nodes,

$$
\text{peak}(x) = \frac{1}{\text{wsum}} \int_0^\alpha W(v)\,P(x,v)\,dv
\;\approx\; \frac{\texttt{np.trapezoid}\big(W_k\,P_k(x),\, v\big)}{\text{wsum}},
$$

and the full two-peak model (7 free parameters) is

$$
\texttt{model}(x) = \text{peak}_\text{left}(x;A_1,c_1,\gamma_1,R_\text{left})
                   + \text{peak}_\text{right}(x;A_2,c_2,\gamma_2,R_\text{right})
                   + \text{offset}.
$$

Free parameter vector (identical layout to the plain 2-Lorentzian):

```
[amp1, cen1, gamma1, amp2, cen2, gamma2, offset]
```

`cen1 = c_left`, `cen2 = c_right` are the **true 180° positions in pixels**.

---

## 4. Where the fixed inputs come from

### Aperture half-angle `α` (both models)
From the objective NA and sample index via Snell (`spectrum_fitter.py`):

$$
\alpha = \arcsin\!\left(\frac{\texttt{na\_collection}}{\texttt{na\_n\_sample}}\right).
$$

- `na_lorentzian*`: `na_collection` is the **effective** NA (calibrated on
  water; absorbs the coupling apodization since the pupil is treated uniform).
- `na_gauss_lorentzian*`: `na_collection` is the **nominal** objective NA (the
  physical pupil edge); the apodization is modeled explicitly (below).

### Gaussian coupling width `v_0` (gauss models only)
From the beam/objective geometry (`na_correction5.gaussian_angle_width`), the
1/e² fiber-mode diameter `D` at the pupil and objective focal length `f`:

$$
v_0 = \arcsin\!\left(\frac{\sin\big(\arctan\!\frac{D/2}{f}\big)}{\texttt{na\_n\_sample}}\right).
$$

Config fields: `na_beam_diameter_mm` = `D` (the session-calibrated knob,
shared by both objectives), `na_focal_length_mm` = `f` (10 mm for 20X, 40 mm
for 5X). `arctan((D/2)/f)` is the beam half-angle in air; Snell maps it into
the sample.

### Elastic anchors `R_left`, `R_right`
From the calibration polynomials, `CalibrationCalculator.elastic_anchors()`.
The per-peak calibration maps pixel → frequency, `f_side(x) = polyval(coeffs, x)`.
The elastic line is the **zero-frequency root**, found by Newton iteration

$$
x \leftarrow x - \frac{f_\text{side}(x)}{f_\text{side}'(x)},
$$

seeded from the calibration sideband point with the smallest `|f|` (the sweep
covers ~4–8 GHz, so 0 GHz is an extrapolation). Linear calibrations solve
directly. Missing/invalid polys or non-convergence **raise** (no fallback).

---

## 5. Selecting & running the fit

Config (`find_peaks_config.toml`, `[sample]`), fields on `FindPeaksConfig`:

| field | `na_lorentzian*` | `na_gauss_lorentzian*` |
|---|---|---|
| `fitting_model` | `na_lorentzian` / `na_lorentzian_window` | `na_gauss_lorentzian` / `na_gauss_lorentzian_window` |
| `na_collection` | effective NA (water-calibrated) | nominal objective NA (0.14 / 0.42) |
| `na_beam_diameter_mm` | — | `D`, fiber-mode Ø at pupil (water-calibrated) |
| `na_focal_length_mm` | — | objective `f` (40 / 10) |
| `na_n_sample` | sample `n` (1.33 water, 1.376 cornea) | same |

`SpectrumFitter.fit(px, sline, is_reference_mode, anchors)`:

1. Detects two peaks (find_peaks); a single peak → `is_success=False` (the
   anchor pairing needs both).
2. Requires `anchors` (an `ElasticAnchors`) — else raises. Validates the NA
   inputs (`0 < na_collection < n`, and `D,f > 0` for gauss) — else raises.
3. Computes `α` (and `v_0`), builds the model via
   `make_2na_lorentzian_binned(R_left, R_right, α, v0=v_0)`.
4. Initial guesses are ordered left→right so peak 1 ↔ `R_left`, peak 2 ↔
   `R_right`. `curve_fit` (TRF, bounded). If fitted centers **cross**
   (`cen2 < cen1`) the fit is rejected (the anchor pairing broke).
5. `_window` variants restrict the fit to a mask of ±`beta·width` pixels
   around each peak (same masking as `lorentzian_window`).

The backends compute `anchors` only when the selected model needs them
(`model_requires_anchors`), and raise if such a model is chosen with no
calibration loaded.

---

## 6. The value chain (why `cen` *is* the shift)

The fitted `cen1`/`cen2` are stored in the **normal**
`left_peak_center_px` / `right_peak_center_px` fields of `FittedSpectrum`, and
`inter_peak_distance = |cen2 − cen1|`. So the standard calibration mapping
(`compute_freq_shift` with `left`/`right`/`distance`) turns them into GHz with
**no NA-specific branch** — the 180° position flows through the existing
left/right/distance frequency outputs unchanged. There are no parallel
"NA-corrected" fields; the correction lives inside the fitted center.

---

## 7. Limits & sanity checks (tests in `tests/spectrum_fitting/`)

- `α → 0`: kernel collapses to a single Lorentzian at `cen` (→ recovers the
  plain model; `test_zero_na_recovers_plain_lorentzian`).
- Uniform weight is exactly `sin(v)`; passing `v0` only *suppresses* large
  angles (`test_default_weight_is_uniform_no_gaussian`).
- A plain symmetric fit to NA data lands **below** `f_180` by the predicted
  mean downshift `⟨frac⟩·(c−R)`; the NA model recovers `f_180`
  (`test_*_recovers_180_degree_position`, `test_plain_lorentzian_is_biased_low`).
- Gauss vs uniform kernels, each water-anchored, agree on `f_180` to well under
  1 MHz across water/glycerol (documented in the project memory).

---

## 8. One-line summary

$$
\boxed{\;
S(x) = \frac{\text{offset}\cdot\text{wsum} + \sum_{\text{peak}}\int_0^{\alpha}
        W(v)\, A\,\gamma\Big[\arctan\tfrac{x+\frac12-x_\text{sub}(v)}{\gamma}
        - \arctan\tfrac{x-\frac12-x_\text{sub}(v)}{\gamma}\Big]\,dv}
       {\text{wsum}},\quad
x_\text{sub}(v)=c-(c-R)(1-\cos\tfrac v2)\;}
$$

with `W(v)=\sin v` (uniform) or `\sin v\,e^{-2(v/v_0)^2}` (gauss), `α` and `v_0`
from the objective/beam geometry, `R` from the calibration, and
`[A_1,c_1,γ_1,A_2,c_2,γ_2,\text{offset}]` fitted. `c_1,c_2` are the 180° shifts.
