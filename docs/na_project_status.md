# NA correction project — status, findings, and paper strategy

**Last updated: 2026-07-17.** Companion to [`na_lineshape_model.md`](na_lineshape_model.md)
(which documents the *model*; this documents the *method, results, and where the
project stands*). Branch: `high_na_fitting`.

This file is written to be picked up cold — no prior conversation needed.

---

## 0. TL;DR — the decision that matters

- **The cornea data is not a paper.** 3 subjects, no lateral structure resolved
  above noise, ±80 µm depth registration, absolute 35 MHz below Scarcelli.
- **There *is* a paper**, but only if reframed from *"we corrected our cornea
  data"* (reviewers shrug at a 0.23% correction) to **metrology**: the collection
  aperture **and fiber-coupling state** impose a **drifting** bias on absolute
  Brillouin shifts — ~8 MHz session-to-session, against clinical effects of
  50–100 MHz. That is a 10–20% uncontrolled systematic in numbers the field
  publishes to three decimals. **The contribution is the discovery, not the fix.**
- **One experiment closes it:** measure `W(v)` directly (reciprocity / pupil
  imaging) instead of fitting it. See §7.
- **Drop width/viscosity entirely** — our own data killed it (§6).

---

## 1. What the method is (one paragraph)

A high-NA objective collects a *cone* of scattering angles, so the recorded peak
is a weighted superposition of sub-peaks each at `f(v) = f₁₈₀·cos(v/2) ≤ f₁₈₀`.
A symmetric fit therefore lands **below** the true 180° shift. We fit the
superposition directly (`na_gauss_lorentzian_window`), anchored at the elastic
line from the calibration, so the fitted centre **is** `f₁₈₀`. Full model and
equations: [`na_lineshape_model.md`](na_lineshape_model.md). Slides explaining
the fit: [`../publication_figures/na_fitting_slides.pptx`](../publication_figures/na_fitting_slides.pptx)
(3 slides: why a plain fit is biased / the model / the implemented pixel-space equation).

Standard config (matches `plot_cornea_shifts.py` next to the data):
`fitting_model="na_gauss_lorentzian_window"`, `na_collection=0.42` (nominal 20X),
`na_focal_length_mm=10.0`, `na_n_sample=1.376` (cornea), `beta=3.0`,
`wlen_pixels=10`; shift per frame = `distance_ghz` (averages both orders).
`na_beam_diameter_mm = D` is **the per-session knob**, calibrated on water.

---

## 2. The strongest results (the paper's spine)

| result | number | why it matters |
|---|---|---|
| **Three-lens collapse** (2026-7-13, water, f10/f25/f40 = NA 0.42/0.246/0.14) | raw spread **15.9 MHz → 0.18 MHz** with **one** shared `D = 6.8 mm` | **Overdetermined**: 3 constraints, 1 free parameter. It could have failed and didn't. **This is the money plot.** |
| **Refractive-index transfer** at fixed calibration | residuals **≲1.4 MHz** across n = 1.33 / 1.355 / 1.376 | the model isn't tuned per material |
| **Kernel-shape robustness** | uniform vs Gauss agree to **~0.5 MHz** | the answer doesn't hinge on the assumed collection profile |
| **Coupling drift (the actual finding)** | `D` wanders **5.17 → 6.8 mm** across sessions (nominal 7.5) ≈ **8 MHz** of absolute shift | an uncontrolled, instrument-state systematic nobody is tracking |

---

## 3. yuxuan2D — analysed 2026-07-17

Data: `…\Data\2026-7-15\afternoon\yuxuan2D.pkl`. Outputs written next to the data:
`yuxuan2D_polar.png`, `analyze_yuxuan2d.py`.

**Structure discovered:** position is encoded in `scan.id` as `R<radius>T<theta>`
(radius mm, theta deg) — a polar grid. 19 spots: R0 ×1, R0.5 ×8, R1.0 ×8,
R1.5 ×2 (partial ring: T0, T270 only). Each spot = **3 frames at the *identical*
lens z** (within-scan spread = 0.000 µm) → pure repeats at ONE depth, **not** a
depth profile.

**Results** (na_gauss_lorentzian_window, **D = 5.78 mm** = this session's *end*
water bracket, the nearest in time; yuxuan2D ran ~10 min before it):

- **Overall: 5.6851 ± 0.0072 GHz** (per-spot medians, n=19)
- per-spot means: 5.6855 ± 0.0063 GHz · all 57 frames: 5.6855 ± 0.0094 GHz
- **0 frames rejected** (all inside the 5.4–6.0 GHz physical window) — clean data
- per-spot median range 5.6727–5.6966 (23.9 MHz)
- Consistent with the yuxuan 1D value (5.683 ± 0.009) — good cross-check.

> **D note:** the notes elsewhere use D = 5.56 (session mean of start 5.35 / end
> 5.78). D is **common-mode**: it slides the whole map by a near-constant offset
> and leaves the spatial pattern and std untouched. **Measured sensitivity:
> ~0.4 MHz per 0.1 mm of D, higher D → higher shift** (slope +3.9 MHz/mm for a
> 20X-only cornea point; computed 2026-07-19 on 7-9 water + 7-15 yuxuan). So over
> a full D span 5.25 → 5.70 the absolute cornea value moves only ~1.75 MHz.
> ⚠️ **CORRECTION:** an earlier draft here said "~+2 MHz per −0.1 mm" — that was
> ~5× too large AND the sign was stated backwards (higher D raises, not lowers,
> the shift); fixed. 5.78 is defensible for yuxuan2D on time-proximity; 5.56 for
> cross-subject consistency. **Not yet standardised — pick one and state it.**

**Per-session water-calibrated D** (na_gauss, self-consistency solve
na042_corr == na014_corr on a two-NA water pair; each session self-calibrates
cleanly, raw two-NA gap −11 to −12 MHz → corrected ≤1.1 MHz):

| session | fitted D [mm] | water f₁₈₀ [GHz] |
|---|---|---|
| 2026-7-9 afternoon | **5.24** | 5.0637 |
| 2026-7-14 water | **5.46** | 5.0676 |
| 2026-7-15 water | **5.67** | 5.0693 |

A clean monotonic rise: **D + 0.43 mm over 6 days** (coupling drift), water f₁₈₀
**+5.6 MHz** (≈ +0.6 °C, warmer water → higher shift — each value is the honest
per-session water shift, not an error). This three-session series *is* the
metrology evidence: the coupling state moves the absolute shift session-to-session,
and one water pair per session removes it. Figures next to each dataset
(`confirm_79.png` / `confirm_714.png` / `confirm_715.png`) and in the weekly deck;
script: session scratchpad `make_confirm_figs.py`.

**No lateral structure is resolved.** This is the honest finding:

- frame-level std **9.4 MHz** → expected spot scatter if the cornea were uniform
  = 9.4/√3 = **5.4 MHz**; observed spot scatter **6.3 MHz** → **excess (real
  spatial) only ~3.3 MHz**.
- Linear gradient fit `shift = a + b·x + c·y`: b = +3.91 ± 1.94 MHz/mm (2.0σ),
  c = +5.03 ± 1.94 MHz/mm (2.6σ); |grad| = 6.37 MHz/mm toward T ≈ 52°; residual
  only 6.3 → 5.2 MHz. **Marginal, post-hoc, and positions are nominal — do not
  claim a gradient.**

**Depth (important):** every measurement sits at `z_offset_um = 200.0` past the
detected anterior surface reflection (`lens_z − surf_z = 200.0` exactly, all 19
scans). Stage travel ≠ tissue depth (focus moves ~n× deeper) → ~275 µm in
tissue ≈ **mid-stroma**, not the anterior plateau.

**Depth registration is the weakest link:** forward vs backward surface
detection disagree by **std ~80 µm** (range −117 to +224 µm). Since depth is
defined relative to that surface, true sampling depth wanders spot-to-spot — and
with a real anterior→posterior gradient, that converts straight into shift noise.
Plausibly explains most of the map's apparent scatter. **Not yet diagnosed:**
whether the reflection finder is *misfiring* (tuning fix) or merely *imprecise*
(hardware limit). Worth pulling peak values / thresholds / DAQ traces.

---

## 4. Scarcelli comparison — resolved, not a contradiction

Paper: Zhang, Asroui, Randleman & Scarcelli, *Biomed. Opt. Express* **13**(12),
6196 (2022), "Motion-tracking Brillouin microscopy for in-vivo corneal
biomechanics mapping". PDF in `…\00_Literature\Brillouin\Scarcelli\`.

**Their setup (from the paper):**
- **780 nm** — *the same laser wavelength as ours*
- **effective NA 0.1** on L3 (f = 50 mm) → their NA bias ≈ **2 MHz, negligible**;
  their number is essentially `f₁₈₀` already
- Rb-locked laser; **0.07% relative accuracy** (~4 MHz), 0.12% stability / 2000 s
- **OCT axial tracking < 3 µm**; 10 µm in-plane pupil tracking
- room temperature **23 °C**; corneal thickness ~500–600 µm
- aqueous humor **5.24 GHz** used as a baseline calibration check

**What they report:** the **mean of the *anterior plateau*** of each depth
profile (plateau end found by intersecting the flat and steep slopes) — *not* a
single depth, *not* a whole-cornea average. Modulus **decreases anterior →
posterior** in their profiles.

**Their values:** inside the **d = 4 mm circle** (i.e. R ≤ 2 mm) ≈ **5.72 GHz**;
periphery *outside* d = 4 mm = 5.74–5.76; superior ~0.02 GHz stiffer than inferior.

**→ Our whole yuxuan2D scan (R ≤ 1.5 mm) lies INSIDE their central zone, so the
comparison value is 5.72, not 5.74–5.76.**

**The gap: ours 5.685 vs their 5.72 = 35 MHz.** Consistent explanation (argued,
**not demonstrated**):
1. **Depth.** We measure one depth at ~mid-stroma; they average the anterior
   plateau, which is the *stiffest* part. The anterior→posterior gradient has the
   right sign and plausible size.
2. Our ±80 µm depth registration means we cannot place ourselves on their profile.

**Not explainable by the NA correction:** reaching 5.72 would need even more
correction, i.e. an even more unphysical D (notes already show yuxuan/zuriel
would need D = 14 mm to hit 5.70).

**DO NOT set `na_collection = 0.12`** to chase their number. It is (a) physically
false — the objective collects 0.42 regardless of what we model; and (b)
**counterproductive**: a smaller modelled aperture *shrinks* the upward
correction, dropping us to ~5.669 (the raw plain-Lorentzian value) — *further*
from 5.72.

---

## 5. Hardware — take, with numbers

**Do not buy or build a 0.12 stop. We already own the low-NA option.**

- **MY5X-802B is NA 0.14** = essentially Scarcelli's regime. Correction there is
  **+1.2 MHz** (vs ~+15 at 20X) → the whole D/coupling systematic evaporates, no
  two-NA water bracket needed.
- **The 5X collects ~2× MORE photons than the 20X** despite 3× less NA — the 20X
  (MY20X-804, designed 436–656 nm) runs **out of band** at 780 nm, so its mode
  overlap has collapsed to s ≈ 0.5. Going low-NA would *gain* signal.
- **Stopping the 20X down to 0.12 is strictly dominated by just using the 5X**:
  same NA regime, but with the out-of-band aberration still present and fewer photons.

**The real cost is axial resolution, which scales as 1/NA² (lateral is 1/NA):**

| | NA | lateral (spec, 550 nm air) | Mitutoyo DOF (550 nm air) | confocal axial, cornea @780 nm |
|---|---|---|---|---|
| 20X | 0.42 | 0.7 µm | 1.6 µm | **~10 µm** |
| 5X | 0.14 | **2.4 µm** | 14 µm | **~96 µm** |
| Scarcelli | 0.10 | — | — | ~190 µm |

> **The 2.4 µm on the 5X spec sheet is LATERAL resolving power** (`0.61λ/NA` at
> 550 nm = 2.4 ✓), **not depth.** Mitutoyo's own DOF specs (1.6 vs 14 µm) already
> encode the **9× ratio** = (0.42/0.14)². The absolute axial number depends on
> convention (air vs tissue, 550 vs 780 nm, DOF vs confocal FWHM) — **the 9×
> ratio does not.**

**Recommendation: fix the *coupling*, not the *aperture*.** Swap
MY20X-804 → **NIR MY20X-824 (NA 0.40)**. Keeps ~10 µm axial resolution *and*
fixes the out-of-band aberration → pushes D toward nominal, onto the **flat top**
of the coupling curve (drift-insensitive) instead of the steep flank at s ≈ 0.5
where D wanders 5.17→6.8. It doesn't remove the ~14 MHz correction; it makes it
**stable and predictable**. Use the 5X as an occasional same-session low-NA
cross-check on the absolute.

**Framing for the PI:** going low-NA to chase Scarcelli's number *concedes our
advantage*. They are at 0.1 because they must be, and their ~190 µm axial
resolution is arguably *why* they can only report a plateau average — some of
their "relatively flat anterior region" may be their instrument, not the tissue.
Our 20X's ~10 µm depth resolution is the genuine differentiator.

---

## 6. Theory / absolute value — and a correction to the record

**Backscattering:** `f_B = 2·n·V / λ₀`, λ₀ = **780 nm**. (This is exactly the
`v → 0` limit of our own NA model: `q(0) = 2nk₀`, `f = Vq/2π`.)

**There is no independent "theoretical" f_B for cornea.** `V` at GHz is precisely
what Brillouin measures. Ultrasound literature V (1550–1640 m/s) predicts
5.47–5.79 GHz — a 320 MHz band, ~45× our precision, useless as a test. (Also the
1636–1640 pachymetry figure is a thickness-calibration constant, not a true
longitudinal sound speed; it predicts *higher* than we measure, the wrong
direction for acoustic dispersion.)

**Inverting our measurement instead** (n = 1.376, yuxuan2D f = 5.6851):
- **V = 1611 m/s**; connor 1614, yuxuan 1611, zuriel 1608
- **M′ = ρV² ≈ 2.76 GPa** (ρ = 1062 kg/m³) — in family with published cornea
  Brillouin moduli

**⚠️ CORRECTION TO AN EARLIER CLAIM (was wrong, now fixed):** the water absolute
check is **NOT** good to 0.1%. An earlier statement compared the 7-15 session's
*implied* temperature against the 7-9 session's *logged* temperature — a mixup.
Done properly:

- **7-9** (the session where **22.4 °C was logged**): f₁₈₀ = 5.0643 → V = 1486.2
  m/s → **implies 21.3 °C**. Water at 22.4 °C should be 1489.5 m/s →
  **3.3 m/s ≈ 11 MHz gap.**
- **7-15**: f₁₈₀ = 5.0745 → V = 1489.2 → implies 22.3 °C (no logged temp available).

→ **Water absolute accuracy ≈ 1 °C / ~10 MHz**, which is *consistent* if 22.4 °C
was room air rather than the cuvette — but it is **not** a 0.1% absolute
validation and must not be presented as one.

**n is the dominant absolute systematic — larger than the NA/coupling one.**
`f_B` is linear in n, so **1% in n = 57 MHz** (8× the spot-to-spot scatter).
Cornea n at 780 nm is arguably **1.371–1.373** (dispersion-corrected) rather than
the 1.376 in our config; that alone moves inferred V by 4–6 m/s. It is
common-mode (doesn't touch relative/spatial results or subject separation) but it
floors any absolute V or M′ we quote.

**Width / viscosity is dead — keep it out of the paper.** Our own data killed it:
NA broadening is only ~1 MHz at these apertures, barely moves with D (residual
6.0→5.2 MHz over D = 4.5→7.5), and the observed broadening didn't replicate
across sessions. This model is a **shift tool only**.

---

## 7. The one experiment that makes it a paper

**Right now `W(v)` is *fitted*, not *measured*.** A reviewer's first question:
*"you added a free parameter and your data fits — so what?"* Today's answer is
the three-lens overdetermination — decent, but defensive.

**Do the reciprocity / pupil-imaging measurement:** back-propagate light through
the collection fiber and image the intensity distribution at the objective pupil.
That turns `W(v)` into a **measured instrument property**, so the NA bias becomes
a **first-principles prediction from an independently measured collection
weight**, not a per-session calibration. Days of lab work, not months. This is
the difference between a note and a paper.

**Other open items:**
- Diagnose the reflection finder (§3) — tuning fix vs hardware limit.
- Standardise D convention (5.56 session-mean vs 5.78 end-bracket) and state it.
- v0-vs-depth has **never** been characterised — required before publishing any
  depth profile (focus-depth aberration could fake a stiffness gradient).
- Water-D vs cornea-D may differ (cornea = curved partial reflector at different
  WD, not a cuvette); no cornea-internal way to check (20X-only).

---

## 8. Paper verdict

- **Cornea paper:** no. Not close.
- **Methods paper framed as "we corrected our cornea data":** weak.
- **Metrology reframe + measured `W(v)`:** yes — a real paper. Modest but genuine
  impact; the field has an acknowledged cross-instrument reproducibility problem.
  Target: *Biomed. Opt. Express* / *J. Biophotonics*.

**The line to give the PI:** the correction is not the contribution — **the
discovery that the fiber-coupling state silently moves the absolute shift by
~8 MHz between sessions is the contribution.** We found it because we built the
tool; most labs cannot see it at all.

**Usability of existing cornea data (unchanged):** genuine and usable for
**relative/spatial** work (NA bias is common-mode and cancels). **Absolute**
carries a one-sided +0/+6 MHz coupling systematic — report with a band, don't
claim 5.70. **Between-subject** differences remain **preliminary**.

---

## 9. Tooling notes (this machine)

- **No `node`, no LibreOffice, no `poppler`/`pdftoppm`.** So: pptxgenjs and the
  pptx skill's soffice/thumbnail scripts do **not** work here.
- **Use the project venv:** `C:\Users\cplan\Documents\repos\brillouin_system\.venv\Scripts\python.exe`
  (has numpy + matplotlib). Always `sys.path.insert(0, r"…\brillouin_system\src")`
  before importing `brillouin_system`.
- `python-pptx` and `pypdf` are **not** in the venv — install with
  `pip install --target <scratch>/libs …` and `sys.path.insert` (keeps the venv clean).
- **⚠️ DO NOT use PowerPoint COM** (`New-Object -ComObject PowerPoint.Application`).
  On Windows it attaches to Connor's **already-running** PowerPoint instance, so
  `$pp.Quit()` closes his live session with unsaved work. (Learned the hard way
  2026-07-20 — he was actively editing his deck.) Build decks with **python-pptx
  only**; verify **figures** as matplotlib PNGs. There is **no** way to preview a
  built `.pptx` on this machine (no COM, no LibreOffice) — build from verified
  components and let Connor open it.
- numpy 2.x: `arr.ptp()` is removed → use `np.ptp(arr)`.
- matplotlib mathtext: no `\bigl`, `\tfrac`, `\dfrac`, or `\frac12` shorthand.

---

## 10. Session 2026-07-20 — per-order split, PSF test, z-axis, precision, all-5 cornea

### Left/right per-order split (the session's main thread)
- **What it is:** for a symmetric sample L−R should be 0; the residual is a
  **per-order calibration (pixel-phase) artifact** — left/right pixel→GHz polys
  aren't perfectly self-consistent. **Distance = midpoint of L and R, so the
  split CANCELS in `distance`** → always report distance, never a single order.
- **NA-independent** (7-9 water, na_gauss D=5.24): 5X vs 20X splits agree within
  each material — water −4.1/−2.7, 5:1 −4.7/−4.6, 2:1 −8.3/−8.6 MHz (SEM 0.3-1.0,
  so 8-17σ real). → it is NOT an aperture effect.
- **NOT universal / tracks calibration quality** (water, corrected, per session):
  7-9 −3.4, **7-14 −13.4** (bad-cal day), 7-15 −3.5, 7-17 pre/post1/post_sel
  −4.3/−5.7/−5.3. The "split grows with shift" seen within 7-9 (−4→−8) is a RED
  HERRING — 7-14 water is biggest at the smallest shift. Don't claim it scales.
- **Cornea 5 subjects** (D=5.6): connor −2.7, **yuxuan +6.8 (3.8σ)**, zuriel +0.8,
  piyush −1.9, selina +3.1. yuxuan is the one real outlier (matches his earlier
  +8 skew). BUT yuxuan also has the widest spatial sampling (laser-position spread
  1.66 mm vs 0.7-1.2 for others; the 5 subject 1D files are NOT strictly
  center-only), so his +6.8 might be partly positional — NOT yet separated
  (radial-correlation test proposed, not run).
- The **±3 MHz "instrument floor"** band I drew on the plots was NOT rigorously
  derived — a round-number stand-in. The principled floor is the **water split
  itself** (water is truly symmetric). Told Connor; he had me drop the band.

### PSF/dual-chain refit — TRIED via a 2dho worktree (option (a)); it does NOT help
- **Worktree:** `C:\Users\cplan\Documents\repos\bs_2dho` on branch `2dho`
  (`git worktree add`). Run with the high_na_fitting venv python + `sys.path.insert`
  to `bs_2dho/src`. **KEPT for now** (Connor's choice).
- **How to run the PSF chain:** `params = calibrate(scan.calibration_data,
  poyfit_degree=2)` → builds `params.psf_variant` (the 7-9/7-15 subject files DO
  carry `calibration_data` frames); `calc = CalibrationCalculator(params)`;
  `anchors = calc.elastic_anchors()`; config model `lorentzian_psf_window`;
  `compute_freq_shift` is chain-aware (fs.calibration_chain = "psf").
- **Result on 7-9 water (8 scans):** PSF chain does NOT fix the split — mean −5.7
  vs Lorentzian −4.0 (no improvement), and scan-to-scan sd **1.4 → 4.6 MHz**
  (triples; range −12.2…+1.3). Confirms Connor's recollection ("measured PSF →
  large variation") and the record's kernel-reconstruction-noise diagnosis
  (n_per_freq=1). The **left PSF is skewed** (Connor's other hypothesis, confirmed
  in the record). Fix = raise n_per_freq (5-10) or pool a session's ePSF kernels;
  neither retroactive on this data. → **distance remains the answer.**

### D-sensitivity — CONFIRMED (supersedes the earlier wrong number)
- Re-fit yuxuan (20X) at D = 5.0…5.9: **slope +3.9 MHz/mm = 0.39 ≈ 0.4 MHz per
  0.1 mm, higher D → higher shift**. Water gap slope +3.4 MHz/mm (consistent;
  correction lives almost entirely in the 20X). Over D 5.25→5.70 the absolute
  moves only ~1.75 MHz. (The earlier "~2 MHz/0.1 mm, sign backwards" was wrong.)

### Axial (zaber-lens) position vs D — 2026-7-10 `water_zaxis_steps.pkl`
- 4 z positions over 23 mm, matched NA014/NA042 pair at each; **first (13.98 mm)
  and last (14.00 mm) repeat the same z as a drift control**.
- **Drift control:** same z, 18 min apart → D +0.075 mm (the noise floor).
- **14 → 25 mm: no effect** — D = 5.47/5.55/5.52 mm (change < the time drift).
- **z = 2 mm differs:** D = 5.10 mm (−0.4 mm), raw NA gap −10.3 vs −11.5…−11.7.
  Too large to be drift. But it's a **single measurement (no repeat there)**.
- **Verdict:** within the 9-16 mm working range D is flat → per-session water D is
  not corrupted by axial position; only a large excursion (2 mm) moves it (~1.6
  MHz common-mode = 0.4 mm × 0.4 MHz/0.1 mm). Sharpens the earlier "lens-z is
  sub-dominant" note.

### Precision / photon budget — the system is SHOT-NOISE LIMITED
- Repo code checked: `calculate_photon_counts.py` `N = π·A·γ` (γ = HWHM, correct);
  count→electron = `preamp_gain` (EM gain 0, camera in **Conventional** mode →
  correct). `theoretical_precision` terms 1 (`s²/N`) and 2 (`a²/12N`) match
  Thompson; **term 3 (background) differs from the cited Thompson eq** (looks like
  a 1D analog — verify against source; inactive here since bg=None).
- **Measured std / theoretical shot-noise std = 0.86-1.08** across all 6
  material×NA on 7-9 (7× photon range 21k→2.6k). → shot-noise-limited; no excess
  noise. Ratios ~0.92 because `s = HWHM` is used as the Gaussian σ (÷1.177 → ~1.08).
  **One number to verify: e⁻/count (assumed 1.0)** — if it's ~4.5, you'd be ~2×
  above shot noise. 20X ≈ 2× noisier than 5X (photon-starved, ~2.5-3× fewer photons).

### σ_D variance formula (came up while checking Connor's stats code)
- **D is a weighted average of L and R** (all three polys estimate the same ν, so
  weights sum to 1): `w_L = −a_D/a_L`, `w_R = a_D/a_R`. Measured **w_L≈0.556,
  w_R≈0.444** (NOT ½,½ — the two orders have different dispersion: a_L≈+0.28,
  a_R≈−0.35, a_D≈−0.155 GHz/px, opposite signs).
- `σ_D = √(w_L²σ_L² + w_R²σ_R² + 2 w_L w_R ρ σ_L σ_R)`; the common `/2` form is the
  w=½ approximation (Connor uses it, "easier", ~3% off here).
- **ρ(L,R) ≈ −0.1 to −0.24** (anti-correlated) — a common-mode pixel drift shifts
  L and R in OPPOSITE freq directions (opposite slopes) and CANCELS in distance →
  the mechanistic reason distance is the robust observable. Connor's
  `cov_corr_from_pairs` (np.cov ddof=1, corr=cov/σσ) is **correct**.

### All-5 cornea with PER-SESSION D (the recommended presentation)
- **Recommendation given: per-session D = mean of that session's pre/post water
  brackets** (NOT one global D — the two sessions' coupling differs). 7-15 D=5.58
  (start 5.30 / end 5.85), piyush D=6.26 (pre 6.27/post1 6.24), selina D=6.04
  (post1 6.24/post_sel 5.85).
- **With frames < 5.6 GHz removed** (Connor's call — "unclear fits"):
  connor **5.6933±5.2**, yuxuan **5.6818±6.0**, zuriel **5.6765±10.8**, piyush
  **5.6755±5.0**, selina **5.6908±13.5** (mean±std MHz). Raw→corrected = +13 to
  +16 MHz (scales with D; std unchanged raw vs corrected = correction is
  common-mode ✓).
- **DECISION POINT flagged to Connor:** the <5.6 cut removed selina's two low
  LASIK-consistent spots and flipped her from lowest (5.682) to among the highest
  (5.691). Are those bad fits or real treated-zone tissue? (zuriel's std 22.5→10.8
  from the cut looks like genuinely bad frames — that one's clearly right.)

### BUG caught this session (don't repeat)
- A `calib_D` used `fit_scan(...)[0].mean()` → took only the **first frame** of
  each scan (`[0]`), giving nonsense D (7-15 = 7.4 mm, above nominal). Connor's
  skepticism about the high value surfaced it. Fixed to `.mean()`. Lesson: sanity-
  check D against the physical range (nominal 7.5, seen 5.1-6.8), and mean the
  full frame array.

### Slides produced this session (all in `03_weekly_updates/`)
- Main deck: `2026-7-21_NA-fitting_Brillouin-results_with_confirmation.pptx`
  (10 slides; 4-7 = confirmation: 7-9 triplet / 7-14 / 7-15 / summary).
- Standalone `EXTRA_*.pptx` (each a 1-slide copy of the MGB template, drag-in):
  `water_anchors`, `zaxis`, `cornea_LR`, `LR_table`, `cornea_5subjects` (+ mean/std
  table), `cornea_raw_vs_corrected`. All built python-pptx-only; figures verified
  as PNGs, **the .pptx themselves were NOT previewed** (no COM). Analysis scripts
  in the session scratchpad (`…\scratchpad\na_slide\`).
