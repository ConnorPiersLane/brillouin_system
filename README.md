# Brillouin System

Control, acquisition, and analysis software for two confocal Brillouin spectroscopy setups:

- **Human Interface (HI)** — an in-vivo system for measuring the human eye. Combines the Brillouin spectrometer with a stereo-camera eye tracker, pupil detection, laser positioning on the cornea, and safety shutters.
- **Microscope** — a benchtop Brillouin microscope with motorized stage, LED illumination, and a FLIR brightfield camera.

Both systems share the same core: an Andor iXon Ultra EMCCD records the spectrometer output, spectra are fitted live (Lorentzian/Voigt models), and a microwave-driven EOM provides the frequency calibration that converts pixel positions to GHz.

## Background

This repository builds upon and extends two original projects by [Amira Eltony](https://github.com/aeltony):

- [microscope-brillouin-daq](https://github.com/aeltony/microscope-brillouin-daq)
- [macro-brillouin-daq](https://github.com/aeltony/macro-brillouin-daq)

It combines and further develops key features from both repositories.

## Repository layout

```
src/
├── brillouin_system/           # main package
│   ├── guis/
│   │   ├── human_interface/    # HI GUI (hi_frontend / hi_backend / hi_signaller)
│   │   │   └── calibrate_dual_allied_vision_cameras/   # camera calibration tools
│   │   ├── microscope/         # microscope GUI (frontend / backend / signaller)
│   │   └── data_analyzer/      # axial scan viewer + Excel export
│   ├── devices/
│   │   ├── cameras/andor/      # iXon Ultra EMCCD (spectrometer camera, via pylablib)
│   │   ├── cameras/allied/     # dual Allied Vision cameras (eye tracker)
│   │   ├── cameras/flir/       # FLIR brightfield camera (microscope)
│   │   ├── ni/                 # NI-6008 DAQ (reflection detection)
│   │   └── zaber_engines/      # Zaber stages (HI 3-axis, eye lens, microscope)
│   ├── eye_tracker/            # stereo imaging, pupil fitting, cornea position
│   │   ├── stereo_imaging/     # stereo calibration + triangulation + SE3 transforms
│   │   └── calibrate_camera_laser_position/  # laser XY offset calibration
│   ├── spectrum_fitting/       # Lorentzian/Voigt fitting, NA correction, analysis
│   ├── calibration/            # spectrometer px↔GHz calibration (EOM sidebands)
│   ├── scan_managers/          # real-time reflection finder (axial scans)
│   ├── saving_and_loading/     # HDF5 save/load of measurement dataclasses
│   ├── logging_utils/          # multiprocess logging + Qt log bridge
│   └── my_dataclasses/         # shared measurement/result dataclasses
├── brillouin_analysis/         # offline analysis (calibration, 2D plots, polarization)
└── simple_plotting/            # ad-hoc plotting scripts
tests/                          # pytest suite (cameras, devices, fitting, save/load)
```

Most subsystems are configured through TOML files that live next to the code that uses them (e.g. `calibration/config/calibration_config.toml`, `spectrum_fitting/peak_fitting_config/find_peaks_config.toml`, camera configs). The GUIs expose dialogs for editing most of them.

## Entry points

| Application | Run | Notes |
|---|---|---|
| Human Interface GUI | `python src/brillouin_system/guis/human_interface/hi_frontend.py` | Main in-vivo system. `HiBackend(use_dummy=True)` runs without hardware. |
| Microscope GUI | `python src/brillouin_system/guis/microscope/microscope_frontend.py` | Currently wired with dummy devices at the top of the file for testing. |
| Axial scan viewer | `python src/brillouin_system/guis/data_analyzer/axial_scan_manager.py` | Browse/analyze saved axial scans, export to Excel. |
| Camera calibration tools | see [Camera Calibration](#camera-calibration-eye-tracker--human-interface) | Checkerboard capture, stereo calibration, transform fitting. |

There is no `pyproject.toml`/`setup.py` — the code is imported as `brillouin_system.*`, so `src/` must be on the Python path (mark `src` as *Sources Root* in PyCharm, or set `PYTHONPATH=src`).

## Hardware

| Device | Role | Driver / library |
|---|---|---|
| Andor iXon Ultra EMCCD | Spectrometer camera (both systems) | Andor SDK2 via `pylablib` |
| 2× Allied Vision cameras | Stereo eye tracker (HI) | Vimba / `vimbapython` |
| FLIR camera | Brightfield imaging (microscope) | Spinnaker / PySpin |
| Zaber stages | HI 3-axis rig + eye-lens axis; microscope stage | `zaber-motion` |
| NI USB-6008 | Photodiode readout for reflection finding | NI-DAQmx / `nidaqmx` |
| Microwave generator | EOM drive for frequency calibration | `pyserial` |
| Phidget relays | Shutter control | `Phidget22` |

# Installation

Python 3.10 is currently required (PySpin constraint).

### 1. Allied Vision Vimba (Microscope and Human Interface)

1. **Download:**
   Visit the [Allied Vision Vimba download page](https://www.alliedvision.com/en/products/software/vimba/) to obtain the latest version of the Vimba software for your operating system.

2. **Install:**
   Run the installer and follow the on-screen instructions. By default, Vimba is installed in:
   - **Windows:** `C:\Program Files\Allied Vision\Vimba_6.0`
   - *(Installation paths may vary on different operating systems.)*

3. **Install the Vimba-Python package.** After installing Vimba, the `vimbapython` package source is in:

   `C:\Program Files\Allied Vision\Vimba_6.0\VimbaPython\Source`

   **Step A: Copy the package to a writable location** (Program Files is permission-restricted), e.g.:

   `C:\Users\<YourUsername>\VimbaPython\Source`

   For example: `C:\Users\Mandelstam\Documents\Connor\VimbaPython\Source`

   **Step B: Install the package:**

   `pip install "vimbapython @ file:///C:/Users/<YourUsername>/vimba-python/Source"`

   For example: `pip install vimbapython@file:///C:/Users/Mandelstam/Documents/Connor/VimbaPython/Source`

### 2. Spinnaker SDK and PySpin for the FLIR camera (microscope only)

Install the Spinnaker PySpin package.
(Attention: `pip install PySpin` installs *pyspin*, which has nothing to do with PySpin.)
Requirement (currently): Python 3.10.
For example, download the Python 3.10 Spinnaker wheel and install it into the venv:

`pip install "C:\Users\Mandelstam\Downloads\spinnaker_python-4.2.0.83-cp310-cp310-win_amd64\spinnaker_python-4.2.0.83-cp310-cp310-win_amd64.whl"`

### 3. Andor SDK (spectrometer camera)

The iXon Ultra is driven through `pylablib`'s `AndorSDK2Camera`, which needs the Andor SDK2 DLLs on the system (installed with Andor SOLIS or the standalone Andor SDK).

### 4. NI-DAQmx (Human Interface)

Install the NI-DAQmx driver from National Instruments, plus the Python bindings:

`pip install nidaqmx`

### 5. Remaining packages

`pip install -r requirements.txt`

# Spectrometer calibration (px ↔ GHz)

Both GUIs include a calibration routine (`calibration/calibration.py`): the microwave generator steps the EOM drive frequency, producing sidebands at known GHz offsets; each step is recorded and fitted, and the resulting pixel-vs-frequency points give interpolation curves (left peak, right peak, peak distance) used by the spectrum fitter to report shifts and widths in GHz. Calibration parameters are in `calibration/config/calibration_config.toml` (editable from the GUI); calibration data is saved with each measurement and can be inspected via the calibration image dialog.

# Camera Calibration (Eye Tracker / Human Interface)

The eye tracker needs three calibration products. They build on each other, so do them in this order:

| # | Calibration | Output file(s) | Loaded at runtime by |
|---|-------------|----------------|----------------------|
| 1 | Stereo checkerboard (intrinsics + extrinsics) | `calibration_left.json`, `calibration_right.json`, `calibration_stereo.json` | `eye_tracker/stereo_imaging/init_stereo_cameras.py` |
| 2 | Camera → Zaber transform | `left_to_zaber.json` | `eye_tracker/stereo_imaging/init_se3.py` (as `left_to_ref`) |
| 3 | Laser XY position offset | `calibrate_camera_laser_position/offset.toml` | `guis/human_interface/hi_backend.py` |

All GUI tools live in `src/brillouin_system/guis/human_interface/calibrate_dual_allied_vision_cameras/`.
The final config files for steps 1–2 must end up in `src/brillouin_system/eye_tracker/stereo_imaging/stereo_configs/` (they are loaded from there at import time).

## 1. Stereo checkerboard calibration

### Step 1a — Capture checkerboard image pairs

Run `checkerboard_capture_gui.py`. It streams both Allied Vision cameras (via `DualCameraProxy`), shows a live corner-detection overlay, and saves synchronized pairs.

- Choose a save folder, then press **Save Pair** (or **Space**) for each pose.
- Images are written as `<folder>/left/<base>_left.png` and `<folder>/right/<base>_right.png`, plus a `metadata.csv`.
- Move the checkerboard through the volume: different distances, tilts, and positions (including image corners). Aim for ~20+ pairs where **both** cameras see the full board.
- Existing capture sets are under `eye_tracker/stereo_imaging/images/` (e.g. `checkerboards1/`).

### Step 1b — Run the calibration

Run `calibrate_stereo_cameras.py` (Tkinter GUI):

1. **Browse** to the capture folder (it auto-detects the `left/`/`right/` subfolder layout).
2. Enter the board parameters:
   - **Cols / Rows** = number of *inner corners* (squares − 1 per side), not the number of squares.
   - **Square (mm)** = physical square edge length. This sets the metric scale of everything downstream (baseline, triangulation, transform) — get it right.
   - **Model**: `pinhole` (default; `fisheye` also supported).
3. **Scan & Validate Frames** — detects corners in every pair once (same detector as the capture GUI's live overlay) and fills the pair table: detection status, board area %, tilt. Coverage hints are printed to the log (missing image regions, too few tilted views, missing close-ups).
4. **Include/exclude pairs** — every pair has a Use checkbox in the table (click it, or select a row and press Space). Excluded pairs are skipped by all calibration steps, so you can drop bad frames and re-run without recapturing.
5. **Run Left Mono**, then **Run Right Mono** — computes intrinsics (K, dist) per camera from the included frames and writes `<prefix>_left.json` / `<prefix>_right.json` into the image folder. Per-view reprojection errors appear in the table; frames above 1 px are marked red — exclude them and re-run.
6. **Run Stereo (Extrinsics Only)** — solves R, T between the cameras with intrinsics held fixed (`cv2.stereoCalibrate` + `CALIB_FIX_INTRINSIC`) and writes `<prefix>_stereo.json`. Convention: RIGHT camera w.r.t. LEFT; LEFT is the reference/world frame. The log reports the stereo RMS (aim for well under 1 px) and the baseline — sanity-check it against the real camera separation.

### Step 1c — Install the result

Copy the three JSONs into `src/brillouin_system/eye_tracker/stereo_imaging/stereo_configs/` with these exact names:

```
calibration_left.json
calibration_right.json
calibration_stereo.json
```

On the next start, `init_stereo_cameras.py` builds the `StereoCameras` object from them and logs the baseline (sanity-check that it matches the real camera separation in mm).

## 2. Camera → Zaber (rig) coordinate transform

This fits a rigid SE3 transform `T_left_to_zaber` (Umeyama/Kabsch, in `fit_coordinate_system.py`) from 3D point correspondences: points triangulated by the stereo pair in the LEFT-camera frame (mm) vs. the same physical points in Zaber stage coordinates. It requires a valid stereo calibration (step 1).

**Rig-frame convention** (must match the runtime): the frame moves with the stage (cameras + laser ride on it); x = y = 0 on the laser beam axis; z is anchored to the **eye-lens scale** — the laser focus sits at z = (lens reading in mm), so a dot the laser is focused on is at z = lens position. A dot fixed in the room has rig coordinates `x,y = −Δstage`, `z = lens_ref − Δstage_z` (mm, relative to the reference pose) — moving the stage +5 mm makes the dot appear at −5 mm.

**Main workflow — live capture** (`point_capture_gui_coordinates_only.py`):

1. Print/mount a small black dot (fixed in the room), **Start** the cameras. The dot is detected live (subpixel blob detector) and triangulated; the overlay shows the LEFT-frame position and the triangulation RMS in px (orange warning if the two cameras likely see different blobs).
2. **Connect** the eye lens (COM5) and the stage (COM6) — both attach **without homing**, they never move on connect. Park the lens at its working position (target-µm field + **Move Lens**, e.g. 12000) and leave it there for the whole calibration (the GUI warns if it moves after the reference is set, and asks for confirmation before any lens move once a reference exists).
3. Jog the stage (frontend-style ± buttons per axis) until the laser hits the dot and is **focused** on it, then press **Set Reference Pose** — the dot is now at rig (0, 0, lens reading in mm) by definition; the lens reading is captured automatically.
4. Jog to a new pose, hold still ~2 s, press **Capture** (or Space). Rig coordinates are computed automatically from the stage position; the capture averages the last 2 s of detections (median) to suppress jitter. If an image folder is chosen, each capture also saves `left/`, `right/` PNGs and `coordinates/<base>.txt` — the exact layout `calibrate_transformation.py` reads for offline re-processing.
5. The transform is re-fitted after every capture: N, fit RMS, per-point residuals (worst in red), a **scale check** (far from 1.0 = units mismatch), geometry hints (too few points, small spans, coplanar/collinear sets), and a **motion check** comparing camera vs stage displacement between captures (catches wrong-axis moves, sign flips, slipped dot). **Show 3D View** opens a live plot of the rig frame, the stage points, the fitted camera points and residual lines. The **Overlay ZABER coords** checkbox (or **Load Transform JSON**) displays the dot's rig coordinates live on the images.
6. Bad points can be unchecked (Use column) or removed; the fit updates immediately. Every change is auto-saved to a session file (including the reference pose), restorable via **Load Session**.
7. **Install as active** writes `stereo_configs/left_to_zaber.json` directly, backing up the previous file with a timestamp. Restart the HI GUI afterwards (the transform is loaded at import).

Aim for 8–15 points spread over the full working volume, including z variation — a minimal cross of points gives a transform that extrapolates poorly.

**Offline alternative:** `calibrate_transformation.py` fits from a saved dataset (folder with `left/`, `right/`, `coordinates/<base>.txt`).

`init_se3.py` loads the installed JSON as `left_to_ref`, which the `PupilDetector` / eye tracker uses to express triangulated pupil positions in stage coordinates.

## 3. Laser XY position calibration

This measures the offset between the camera-estimated pupil center and the actual laser axis, in Zaber coordinates. It is fully automated and run from the Human Interface backend: `run_laser_xy_calibration()` in `hi_backend.py` (there is a button for it in the HI GUI). It must be run in **sample mode**, not reference mode, with the eye tracker positioned on the calibration rig so the cameras report a pupil center.

What it does (`eye_tracker/calibrate_camera_laser_position/calib_rig_laser_position.py`):

1. Starts at the camera-estimated pupil-center position.
2. Moves outward along rays every `dphi_deg` (default 10°) and uses the NI axial reflection finder to detect where the laser starts hitting the reflection plane of the rig.
3. Refines each boundary crossing with a binary search down to `tolerance_um` (default 5 µm).
4. Fits a 3D circle through all boundary points; the circle center is where the laser actually is.
5. The offset (circle center − camera pupil-center estimate) is saved as `offset.toml` next to the script; the raw boundary points go to `measured_circle.json` (visualize with `show_measurement_results.py`).

Scan parameters (step size, angular resolution, tolerances) are in `calibrate_camera_laser_position/settings.toml`.

Re-run this whenever the cameras or the laser path have been touched; re-run steps 1–2 whenever the cameras have been moved relative to each other or refocused.

# Data, analysis, and tests

- **Saving:** measurements (axial scans, calibration data, spectra) are dataclasses serialized to HDF5 via `saving_and_loading/safe_and_load_hdf5.py`.
- **Viewing:** `guis/data_analyzer/` provides an axial scan viewer (`axial_scan_manager.py` / `show_axial_scan.py`) and Excel export.
- **Offline analysis:** `src/brillouin_analysis/` holds standalone analysis (calibration studies, 2D Brillouin maps, polarization); `src/simple_plotting/` holds one-off plotting scripts.
- **Tests:** run `pytest tests/` — covers cameras, devices, spectrum fitting, eye tracking, and HDF5 round-trips (hardware-dependent tests use the dummy devices).
