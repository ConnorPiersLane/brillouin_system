from __future__ import annotations

"""
Point Capture GUI — optimized

New features
------------
1) Coordinate presets via ComboBox (save/load/select stage coordinates).
2) "Set from preset" button fills (x,y,z) from the selected preset.
3) Collect 3D ↔ 3D correspondences (LEFT camera frame ↔ ZABER stage).
4) Fit rigid (or similarity) transform with robust trimming (Umeyama/Kabsch).
5) Live prediction of the dot location in the selected frame (LEFT/ZABER) once a transform is fitted/loaded.
6) Save/Load the fitted transform as JSON.
7) Optional stereo triangulation to obtain the LEFT-frame 3D point from pixel detections
   using saved calibration JSONs (left.json, right.json, stereo.json).

Dependencies: tkinter, numpy, opencv-python, pillow
"""

import os
import csv
import json
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# ----- Local numeric tools (fit & transforms) -----
# If your project already provides these, you can swap imports accordingly.
try:
    from brillouin_system.eye_tracker.stereo_calibration.fit_coordinate_system import fit_coordinate_system  # returns (SE3, info)
    from brillouin_system.eye_tracker.stereo_calibration.coord_transformer import SE3, CoordTransformer
except Exception:
    raise RuntimeError("Required modules 'fit_coordinate_system' and 'coord_transformer' not found.")


# ----------------- detection helper -----------------

def detect_dot_centroid(gray: np.ndarray, min_area: int) -> Optional[Tuple[float, float, float]]:
    """
    Detect a single dark dot on light background.
    Returns (cx, cy, area) in pixel units or None if not found.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 3)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = [c for c in cnts if cv2.contourArea(c) >= float(min_area)]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < float(min_area):
        return None
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    # Subpixel refine
    pt = np.array([[cx, cy]], np.float32)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    cv2.cornerSubPix(gray, pt, (5, 5), (-1, -1), term)
    return float(pt[0, 0]), float(pt[0, 1]), area


# ----------------- file pairing helper -----------------

def load_image_pairs_simple(folder: str) -> List[Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Pair images by basename suffixes _left/_right or -left/-right.
    Returns list of (img_left_bgr, img_right_bgr, path_left, path_right).
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    stems: Dict[str, Dict[str, str]] = {}
    for f in files:
        name, _ = os.path.splitext(f)
        if name.endswith("_left") or name.endswith("-left"):
            stem = name[:-5]
            stems.setdefault(stem, {})["left"] = os.path.join(folder, f)
        elif name.endswith("_right") or name.endswith("-right"):
            stem = name[:-6]
            stems.setdefault(stem, {})["right"] = os.path.join(folder, f)

    pairs: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
    for stem, sides in sorted(stems.items()):
        if "left" in sides and "right" in sides:
            L = cv2.imread(sides["left"], cv2.IMREAD_COLOR)
            R = cv2.imread(sides["right"], cv2.IMREAD_COLOR)
            if L is None or R is None:
                continue
            pairs.append((L, R, sides["left"], sides["right"]))
    return pairs


# ----------------- stereo triangulation (JSON-based) -----------------

@dataclass
class StereoModel:
    K_left: np.ndarray
    d_left: Optional[np.ndarray]
    K_right: np.ndarray
    d_right: Optional[np.ndarray]
    R_lr: np.ndarray  # RIGHT w.r.t LEFT
    T_lr: np.ndarray
    reference: str  # 'left' or 'right'
    image_size: Optional[Tuple[int, int]]

    @staticmethod
    def from_json_files(path_left_json: str, path_right_json: str, path_stereo_json: str) -> "StereoModel":
        with open(path_left_json, "r", encoding="utf-8") as f:
            L = json.load(f)
        with open(path_right_json, "r", encoding="utf-8") as f:
            R = json.load(f)
        with open(path_stereo_json, "r", encoding="utf-8") as f:
            S = json.load(f)
        K_left = np.array(L["camera"]["K"], dtype=float)
        d_left = np.array(L["camera"].get("dist", []), dtype=float) if L["camera"].get("dist") is not None else None
        K_right = np.array(R["camera"]["K"], dtype=float)
        d_right = np.array(R["camera"].get("dist", []), dtype=float) if R["camera"].get("dist") is not None else None
        R_lr = np.array(S["stereo"]["R"], dtype=float)
        T_lr = np.array(S["stereo"]["T"], dtype=float)
        ref = S["stereo"].get("reference", "left").lower()
        w, h = L.get("image_size", [0, 0])
        img_size = (int(w), int(h)) if w and h else None
        return StereoModel(K_left, d_left, K_right, d_right, R_lr, T_lr, ref, img_size)

    def undistort_points(self, which: str, pts_px: np.ndarray) -> np.ndarray:
        K = self.K_left if which == "left" else self.K_right
        d = self.d_left if which == "left" else self.d_right
        pts_px = np.asarray(pts_px, dtype=float).reshape(-1, 1, 2)
        if d is None or d.size == 0:
            invK = np.linalg.inv(K)
            uv1 = np.concatenate([pts_px.reshape(-1, 2), np.ones((len(pts_px), 1))], axis=1)
            rays = (invK @ uv1.T).T
            rays /= rays[:, 2:3]
            return rays[:, :2]
        norm = cv2.undistortPoints(pts_px, K, d, P=None)
        return norm.reshape(-1, 2)

    def _poses_left_world(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (R_left, t_left, R_right, t_right) in a WORLD where LEFT is origin.
        If reference is 'right', invert R_lr/T_lr accordingly.
        """
        if self.reference == "left":
            R_left = np.eye(3); t_left = np.zeros(3)
            R_right = self.R_lr.copy(); t_right = self.T_lr.copy()
        else:
            # RIGHT is reference in the file → invert RIGHT→LEFT
            R_rl = self.R_lr.T
            T_rl = -self.R_lr.T @ self.T_lr
            R_left = R_rl.copy(); t_left = T_rl.copy()
            R_right = np.eye(3); t_right = np.zeros(3)
        return R_left, t_left, R_right, t_right

    def triangulate_midpoint(self, uvL: Tuple[float, float], uvR: Tuple[float, float]) -> np.ndarray:
        """
        Geometric midpoint of the shortest segment between the two viewing rays.
        Returns X in the LEFT-world coordinates (3,).
        """
        Rl, tl, Rr, tr = self._poses_left_world()
        # Undistort to normalized camera plane
        nL = self.undistort_points("left", np.array([uvL]))[0]
        nR = self.undistort_points("right", np.array([uvR]))[0]
        d1 = (Rl.T @ np.array([nL[0], nL[1], 1.0]))  # left cam ray in LEFT-world
        d2 = (Rr.T @ np.array([nR[0], nR[1], 1.0]))  # right cam ray in LEFT-world
        d1 /= np.linalg.norm(d1); d2 /= np.linalg.norm(d2)
        C1 = -Rl.T @ tl
        C2 = -Rr.T @ tr
        r = C2 - C1
        a = float(d1 @ d1); b = float(d1 @ d2); c = float(d2 @ d2)
        d = float(d1 @ r);  e = float(d2 @ r)
        denom = a * c - b * b
        if abs(denom) < 1e-12:
            s = d / max(a, 1e-12)
            return C1 + s * d1
        s = (d * c - b * e) / denom
        t = (a * e - b * d) / denom
        P1 = C1 + s * d1
        P2 = C2 + t * d2
        return 0.5 * (P1 + P2)


# ----------------- App state -----------------

@dataclass
class AppState:
    folder: str = ""
    min_area: int = 200

    # image pairs & per-pair detections
    pairs: Optional[List[Tuple[np.ndarray, np.ndarray, str, str]]] = None
    pair_index: int = 0
    det_cache: Optional[List[Tuple[bool, Optional[Tuple[float, float, float]],
                                   bool, Optional[Tuple[float, float, float]]]]] = None

    # user coordinates (µm)
    coords_um: Optional[List[Optional[Tuple[float, float, float]]]] = None

    # stereo model
    stereo: Optional[StereoModel] = None

    # fitted transform LEFT→ZABER (rigid) and optional scale
    T_left_to_zaber: Optional[SE3] = None
    scale: float = 1.0
    fit_info: Optional[dict] = None

    # coordinate presets (Combobox)
    presets: Dict[str, Tuple[float, float, float]] = None

    # display preference
    show_frame: str = "zaber"  # or "left"


# ----------------- GUI -----------------

class PointCaptureGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Point Capture GUI — Optimized")
        self.state = AppState(presets={})
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Folder row
        row = 0
        ttk.Label(root, text="Images Folder:").grid(row=row, column=0, sticky="w")
        self.var_folder = tk.StringVar()
        ttk.Entry(root, textvariable=self.var_folder, width=50).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(root, text="Browse", command=self._on_browse).grid(row=row, column=2)
        root.columnconfigure(1, weight=1)

        # Parameters + Stereo
        row += 1
        p = ttk.Frame(root)
        p.grid(row=row, column=0, columnspan=3, pady=8, sticky="we")
        self.var_min_area = tk.IntVar(value=200)
        ttk.Label(p, text="min_area (px²)").grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(p, textvariable=self.var_min_area, width=8).grid(row=0, column=1, padx=(0, 12))
        ttk.Button(p, text="Load Stereo Calib…", command=self._load_stereo_jsons).grid(row=0, column=2, padx=(0, 12))
        self.lbl_st = ttk.Label(p, text="Stereo: not loaded", foreground="#999")
        self.lbl_st.grid(row=0, column=3)
        p.columnconfigure(3, weight=1)

        # Command buttons
        row += 1
        b = ttk.Frame(root)
        b.grid(row=row, column=0, columnspan=3, pady=6)
        ttk.Button(b, text="Scan Frames", command=lambda: self._async(self._scan_frames_impl)).grid(row=0, column=0, padx=5)
        ttk.Button(b, text="Detect Current", command=self._detect_current_pair).grid(row=0, column=1, padx=5)
        ttk.Button(b, text="Detect All", command=lambda: self._async(self._detect_all_impl)).grid(row=0, column=2, padx=5)
        ttk.Button(b, text="Fit Transform", command=self._fit_transform).grid(row=0, column=3, padx=5)
        ttk.Button(b, text="Save Transform…", command=self._save_transform_json).grid(row=0, column=4, padx=5)
        ttk.Button(b, text="Load Transform…", command=self._load_transform_json).grid(row=0, column=5, padx=5)

        # Fitting options
        row += 1
        opt = ttk.Frame(root)
        opt.grid(row=row, column=0, columnspan=3, pady=(0, 6), sticky="we")
        self.var_with_scale = tk.BooleanVar(value=False)
        self.var_trim = tk.DoubleVar(value=0.0)
        self.var_repeats = tk.IntVar(value=1)
        ttk.Checkbutton(opt, text="Estimate scale", variable=self.var_with_scale).grid(row=0, column=0, padx=(0, 12))
        ttk.Label(opt, text="Trim fraction").grid(row=0, column=1)
        ttk.Entry(opt, textvariable=self.var_trim, width=6).grid(row=0, column=2, padx=(4, 12))
        ttk.Label(opt, text="Repeats").grid(row=0, column=3)
        ttk.Entry(opt, textvariable=self.var_repeats, width=6).grid(row=0, column=4, padx=(4, 12))

        # Viewer + status
        row += 1
        vf = ttk.Frame(root)
        vf.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=8)
        root.rowconfigure(row, weight=1)

        # Viewer controls
        ctrl = ttk.Frame(vf)
        ctrl.grid(row=0, column=0, sticky="we", pady=(0, 6))
        self.btn_prev = ttk.Button(ctrl, text="⟵ Prev", command=self._on_prev_pair, state="disabled")
        self.btn_next = ttk.Button(ctrl, text="Next ⟶", command=self._on_next_pair, state="disabled")
        self.lbl_status = ttk.Label(ctrl, text="No pairs loaded")
        self.lbl_detect = ttk.Label(ctrl, text="• Idle", foreground="#999")
        self.btn_prev.grid(row=0, column=0, padx=5)
        self.btn_next.grid(row=0, column=1, padx=5)
        self.lbl_status.grid(row=0, column=2, padx=10)
        self.lbl_detect.grid(row=0, column=3, padx=10)
        ctrl.columnconfigure(2, weight=1)

        canv = ttk.Frame(vf)
        canv.grid(row=1, column=0, sticky="nsew")
        self.cnv_left = tk.Canvas(canv, width=480, height=360, bg="#111")
        self.cnv_right = tk.Canvas(canv, width=480, height=360, bg="#111")
        self.cnv_left.grid(row=0, column=0, padx=4, pady=4)
        self.cnv_right.grid(row=0, column=1, padx=4, pady=4)
        canv.columnconfigure(0, weight=1)
        canv.columnconfigure(1, weight=1)

        # Coordinate entry & presets
        coord = ttk.Frame(vf)
        coord.grid(row=2, column=0, sticky="we", pady=(8, 0))
        ttk.Label(coord, text="Coordinates (µm):").grid(row=0, column=0, sticky="w")
        self.var_x = tk.StringVar(); self.var_y = tk.StringVar(); self.var_z = tk.StringVar()
        ttk.Label(coord, text="x").grid(row=0, column=1); ttk.Entry(coord, textvariable=self.var_x, width=10).grid(row=0, column=2, padx=4)
        ttk.Label(coord, text="y").grid(row=0, column=3); ttk.Entry(coord, textvariable=self.var_y, width=10).grid(row=0, column=4, padx=4)
        ttk.Label(coord, text="z").grid(row=0, column=5); ttk.Entry(coord, textvariable=self.var_z, width=10).grid(row=0, column=6, padx=4)
        ttk.Button(coord, text="Save coords for this pair", command=self._save_coords_for_pair).grid(row=0, column=7, padx=8)

        # Presets row
        pr = ttk.Frame(vf)
        pr.grid(row=3, column=0, sticky="we", pady=(6, 0))
        ttk.Label(pr, text="Coordinate presets:").grid(row=0, column=0)
        self.combo_preset = ttk.Combobox(pr, width=30, state="readonly", values=[])
        self.combo_preset.grid(row=0, column=1, padx=6)
        ttk.Button(pr, text="Set from preset", command=self._set_coords_from_preset).grid(row=0, column=2, padx=4)
        ttk.Button(pr, text="Add/Update preset…", command=self._add_or_update_preset).grid(row=0, column=3, padx=4)
        ttk.Button(pr, text="Remove preset", command=self._remove_preset).grid(row=0, column=4, padx=4)

        # Display toggle & prediction
        dr = ttk.Frame(vf)
        dr.grid(row=4, column=0, sticky="we", pady=(8, 0))
        ttk.Label(dr, text="Show coordinates in:").grid(row=0, column=0)
        self.var_show_frame = tk.StringVar(value="zaber")
        ttk.Radiobutton(dr, text="ZABER", variable=self.var_show_frame, value="zaber", command=self._render_pair).grid(row=0, column=1)
        ttk.Radiobutton(dr, text="LEFT (camera)", variable=self.var_show_frame, value="left", command=self._render_pair).grid(row=0, column=2)
        self.lbl_pred = ttk.Label(dr, text="Prediction: n/a")
        self.lbl_pred.grid(row=0, column=3, padx=12)
        dr.columnconfigure(3, weight=1)

        # Export buttons
        row += 1
        ex = ttk.Frame(root)
        ex.grid(row=row, column=0, columnspan=3, pady=6)
        ttk.Button(ex, text="Export (CSV)", command=lambda: self._export('csv')).grid(row=0, column=0, padx=5)
        ttk.Button(ex, text="Export (JSON)", command=lambda: self._export('json')).grid(row=0, column=1, padx=5)
        ttk.Button(ex, text="Export (TOML)", command=lambda: self._export('toml')).grid(row=0, column=2, padx=5)

        # Log
        row += 1
        ttk.Label(root, text="Log:").grid(row=row, column=0, sticky="w")
        row += 1
        self.txt_log = tk.Text(root, height=8)
        self.txt_log.grid(row=row, column=0, columnspan=3, sticky="nsew")
        root.rowconfigure(row, weight=1)

    # ---------- helpers ----------
    def _on_browse(self):
        folder = filedialog.askdirectory()
        if folder:
            self.var_folder.set(folder)

    def _async(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    def _flash_detect(self, ok_left: bool, ok_right: bool):
        if ok_left and ok_right:
            self.lbl_detect.configure(text="• Detected (both)", foreground="#16a34a")
        elif ok_left or ok_right:
            self.lbl_detect.configure(text="• Partial detection", foreground="#ca8a04")
        else:
            self.lbl_detect.configure(text="• No detection", foreground="#dc2626")

    # ---------- operations ----------
    def _scan_frames_impl(self):
        folder = self.var_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            self._log("Invalid folder")
            return
        self._log("Scanning for left/right image pairs…")
        pairs = load_image_pairs_simple(folder)
        if not pairs:
            self._log("No pairs found (expect *_left.* and *_right.*)")
            self.state.pairs = None
            self.state.det_cache = None
            self.state.coords_um = None
            self.state.pair_index = 0
            self._render_pair()
            return
        self.state.pairs = pairs
        self._log(f"Found {len(pairs)} pairs")
        self.state.det_cache = [(False, None, False, None) for _ in pairs]
        self.state.coords_um = [None for _ in pairs]
        self.state.pair_index = 0
        self._render_pair()

    def _detect_current_pair(self):
        if not self.state.pairs:
            self._log("No pairs loaded")
            self._flash_detect(False, False)
            return
        i = self.state.pair_index
        L, R, _, _ = self.state.pairs[i]
        min_area = int(self.var_min_area.get())
        resL = detect_dot_centroid(L, min_area)
        resR = detect_dot_centroid(R, min_area)
        okL = resL is not None
        okR = resR is not None
        self.state.det_cache[i] = (okL, resL, okR, resR)
        self._flash_detect(okL, okR)
        self._log(f"Pair {i+1}: left={'OK' if okL else 'FAIL'} right={'OK' if okR else 'FAIL'} (min_area={min_area})")
        self._render_pair()

    def _detect_all_impl(self):
        if not self.state.pairs:
            self._log("No pairs loaded")
            self._flash_detect(False, False)
            return
        min_area = int(self.var_min_area.get())
        self._log(f"Detecting dots on all pairs (min_area={min_area})…")
        cache = []
        for i, (L, R, _, _) in enumerate(self.state.pairs, 1):
            resL = detect_dot_centroid(L, min_area)
            resR = detect_dot_centroid(R, min_area)
            cache.append((resL is not None, resL, resR is not None, resR))
            if i % 10 == 0:
                self._log(f"  processed {i}/{len(self.state.pairs)}")
        self.state.det_cache = cache
        ok_both = sum(1 for okL, _, okR, _ in cache if okL and okR)
        self._log(f"Detection complete. Both-sides detections: {ok_both}/{len(cache)}")
        if cache:
            self._flash_detect(cache[-1][0], cache[-1][2])
        self._render_pair()

    def _save_coords_for_pair(self):
        if not self.state.pairs:
            self._log("No pairs loaded")
            return
        i = self.state.pair_index
        try:
            x = float(self.var_x.get()); y = float(self.var_y.get()); z = float(self.var_z.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric x, y, z in micrometers.")
            return
        self.state.coords_um[i] = (x, y, z)
        self._log(f"Saved coords for pair {i+1}: ({x:.3f}, {y:.3f}, {z:.3f}) µm")
        self._render_pair()

    # ----- Presets -----
    def _set_coords_from_preset(self):
        name = self.combo_preset.get()
        if not name or name not in self.state.presets:
            return
        x, y, z = self.state.presets[name]
        self.var_x.set(str(x)); self.var_y.set(str(y)); self.var_z.set(str(z))
        self._log(f"Loaded preset '{name}' → ({x}, {y}, {z}) µm")

    def _add_or_update_preset(self):
        name = simpledialog.askstring("Preset name", "Enter a name for this coordinate preset:")
        if not name:
            return
        try:
            x = float(self.var_x.get()); y = float(self.var_y.get()); z = float(self.var_z.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric x, y, z first.")
            return
        self.state.presets[name] = (x, y, z)
        self.combo_preset["values"] = sorted(self.state.presets.keys())
        self.combo_preset.set(name)
        self._log(f"Preset saved: {name} → ({x}, {y}, {z}) µm")

    def _remove_preset(self):
        name = self.combo_preset.get()
        if name and name in self.state.presets:
            del self.state.presets[name]
            self.combo_preset["values"] = sorted(self.state.presets.keys())
            self.combo_preset.set("")
            self._log(f"Preset removed: {name}")

    # ----- Stereo calib load -----
    def _load_stereo_jsons(self):
        msg = "Select LEFT camera JSON"
        pL = filedialog.askopenfilename(title=msg, filetypes=[("JSON", "*.json")])
        if not pL:
            return
        pR = filedialog.askopenfilename(title="Select RIGHT camera JSON", filetypes=[("JSON", "*.json")])
        if not pR:
            return
        pS = filedialog.askopenfilename(title="Select STEREO JSON", filetypes=[("JSON", "*.json")])
        if not pS:
            return
        try:
            self.state.stereo = StereoModel.from_json_files(pL, pR, pS)
            self.lbl_st.configure(text=f"Stereo: loaded (ref={self.state.stereo.reference})", foreground="#16a34a")
            self._log("Stereo calibration loaded.")
        except Exception as e:
            messagebox.showerror("Stereo load failed", str(e))
            self.state.stereo = None
            self.lbl_st.configure(text="Stereo: not loaded", foreground="#dc2626")

    # ----- Fitting -----
    def _gather_correspondences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build A (LEFT 3D) and B (ZABER 3D) from current detections and saved coords.
        LEFT 3D is triangulated from (uL,uR) using loaded stereo model.
        """
        if self.state.stereo is None:
            raise RuntimeError("Stereo calibration not loaded.")
        if not self.state.pairs or not self.state.det_cache:
            raise RuntimeError("No detections.")
        A_left: List[List[float]] = []
        B_zaber: List[List[float]] = []
        for (okL, cL, okR, cR), cu in zip(self.state.det_cache, self.state.coords_um or []):
            if not (okL and okR and cL and cR and cu is not None):
                continue
            uL = (cL[0], cL[1]); uR = (cR[0], cR[1])
            Xl = self.state.stereo.triangulate_midpoint(uL, uR)
            A_left.append([float(Xl[0]), float(Xl[1]), float(Xl[2])])
            B_zaber.append([float(cu[0]), float(cu[1]), float(cu[2])])
        if len(A_left) < 3:
            raise RuntimeError("Need at least 3 valid correspondences (both dots detected + coords saved).")
        return np.array(A_left, float), np.array(B_zaber, float)

    def _fit_transform(self):
        try:
            A, B = self._gather_correspondences()
        except Exception as e:
            messagebox.showerror("Fit failed", str(e))
            return
        with_scale = bool(self.var_with_scale.get())
        trim = float(self.var_trim.get())
        reps = int(self.var_repeats.get())
        try:
            T, info = fit_coordinate_system(A, B, with_scale=with_scale, trim_fraction=trim, trim_repeats=reps)
        except Exception as e:
            messagebox.showerror("Fit failed", str(e))
            return
        self.state.T_left_to_zaber = T
        self.state.scale = float(info.get("scale", 1.0))
        self.state.fit_info = info
        rms = info.get("rms", float("nan"))
        nin = info.get("num_inliers", 0)
        self._log(f"Fitted LEFT→ZABER (scale={self.state.scale:.6f}, rms={rms:.4f}, inliers={nin})")
        self._render_pair()

    # ----- Transform save/load -----
    def _save_transform_json(self):
        if self.state.T_left_to_zaber is None:
            messagebox.showwarning("Nothing to save", "Run 'Fit Transform' first or load an existing transform.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="left_to_zaber_transform.json")
        if not out:
            return
        T = self.state.T_left_to_zaber
        info = self.state.fit_info or {}
        payload = {
            "source": "point_capture_gui",
            "timestamp": int(time.time()),
            "frames": {"src": "left", "dst": "zaber"},
            "R": T.R.tolist(),
            "t": T.t.tolist(),
            "scale": float(self.state.scale),
            "fit": {
                "rms": float(info.get("rms", float("nan"))),
                "num_inliers": int(info.get("num_inliers", 0))
            }
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._log(f"Saved transform JSON: {out}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _load_transform_json(self):
        p = filedialog.askopenfilename(title="Select transform JSON", filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            R = np.array(d["R"], dtype=float); t = np.array(d["t"], dtype=float)
            self.state.T_left_to_zaber = SE3(R, t)
            self.state.scale = float(d.get("scale", 1.0))
            self.state.fit_info = d.get("fit", {})
            self._log(f"Loaded transform (scale={self.state.scale:.6f}).")
            self._render_pair()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    # ----- Export detections/coords -----
    def _export(self, fmt: str):
        if not self.state.pairs or not self.state.det_cache:
            self._log("Nothing to export")
            return
        if fmt == 'csv':
            out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="dot_detections.csv")
        elif fmt == 'json':
            out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="dot_detections.json")
        elif fmt == 'toml':
            out = filedialog.asksaveasfilename(defaultextension=".toml", filetypes=[("TOML", "*.toml")], initialfile="dot_detections.toml")
        else:
            return
        if not out:
            return
        data_rows = []
        for i, ((L, R, pL, pR), (okL, cL, okR, cR), coords) in enumerate(zip(self.state.pairs, self.state.det_cache, self.state.coords_um or []), 1):
            row = {
                "index": i,
                "left_path": pL,
                "right_path": pR,
                "left_cx": None if not cL else float(cL[0]),
                "left_cy": None if not cL else float(cL[1]),
                "left_area": None if not cL else float(cL[2]),
                "right_cx": None if not cR else float(cR[0]),
                "right_cy": None if not cR else float(cR[1]),
                "right_area": None if not cR else float(cR[2]),
                "detected_both": bool(okL and okR),
                "coords_um": None if coords is None else {"x": coords[0], "y": coords[1], "z": coords[2]},
            }
            data_rows.append(row)
        try:
            if fmt == 'csv':
                with open(out, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["index", "left_path", "right_path",
                                "left_cx", "left_cy", "left_area",
                                "right_cx", "right_cy", "right_area",
                                "detected_both",
                                "coords_um_x", "coords_um_y", "coords_um_z"])
                    for r in data_rows:
                        cu = r["coords_um"] or {}
                        w.writerow([
                            r["index"], r["left_path"], r["right_path"],
                            r["left_cx"], r["left_cy"], r["left_area"],
                            r["right_cx"], r["right_cy"], r["right_area"],
                            r["detected_both"],
                            cu.get("x", ""), cu.get("y", ""), cu.get("z", ""),
                        ])
            elif fmt == 'json':
                with open(out, "w", encoding="utf-8") as f:
                    json.dump({"rows": data_rows}, f, indent=2)
            elif fmt == 'toml':
                def esc(s: str) -> str:
                    return s.replace('\\', '\\\\').replace('"', '\\"')
                with open(out, "w", encoding="utf-8") as f:
                    f.write('# dot detections export\n')
                    f.write("\n[[rows]]\n")
                    for idx, r in enumerate(data_rows):
                        if idx > 0:
                            f.write("\n[[rows]]\n")
                        f.write(f'index = {r["index"]}\n')
                        f.write(f'left_path = "{esc(r["left_path"])}"\n')
                        f.write(f'right_path = "{esc(r["right_path"])}"\n')
                        for key in ("left_cx","left_cy","left_area","right_cx","right_cy","right_area"):
                            val = r[key]
                            f.write(f'{key} = {"null" if val is None else float(val)}\n')
                        f.write(f'detected_both = {"true" if r["detected_both"] else "false"}\n')
                        cu = r["coords_um"]
                        if cu is None:
                            f.write('coords_um = { x = null, y = null, z = null }\n')
                        else:
                            f.write(f'coords_um = {{ x = {cu["x"]}, y = {cu["y"]}, z = {cu["z"]} }}\n')
            self._log(f"Saved {fmt.upper()} to: {out}")
            messagebox.showinfo("Export", f"Saved {fmt.upper()} to:\n{out}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ---------- viewer ----------
    def _on_prev_pair(self):
        if not self.state.pairs:
            return
        self.state.pair_index = max(0, self.state.pair_index - 1)
        self._render_pair()

    def _on_next_pair(self):
        if not self.state.pairs:
            return
        n = len(self.state.pairs)
        self.state.pair_index = min(n - 1, self.state.pair_index + 1)
        self._render_pair()

    def _render_pair(self):
        pairs = self.state.pairs or []
        if not pairs:
            self.lbl_status.configure(text="No pairs loaded")
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
            self._draw_blank(self.cnv_left); self._draw_blank(self.cnv_right)
            self.lbl_pred.configure(text="Prediction: n/a")
            return
        i = self.state.pair_index
        n = len(pairs)
        self.btn_prev.config(state="normal" if i > 0 else "disabled")
        self.btn_next.config(state="normal" if i < n - 1 else "disabled")
        L, R, pL, pR = pairs[i]
        okL, cL, okR, cR = self.state.det_cache[i] if self.state.det_cache and i < len(self.state.det_cache) else (False, None, False, None)
        both = okL and okR
        self.lbl_status.configure(text=f"Pair {i+1}/{n}  |  left: {'OK' if okL else 'FAIL'}  |  right: {'OK' if okR else 'FAIL'}  |  both: {'OK' if both else 'FAIL'}")

        # Set entries if coords saved
        if self.state.coords_um and self.state.coords_um[i] is not None:
            x, y, z = self.state.coords_um[i]
            self.var_x.set(f"{x}"); self.var_y.set(f"{y}"); self.var_z.set(f"{z}")

        # Draw images
        self._draw_img(self.cnv_left, L, cL)
        self._draw_img(self.cnv_right, R, cR)

        # Prediction label (if transform is known and detections available)
        pred_txt = "Prediction: n/a"
        if self.state.T_left_to_zaber is not None and both and self.state.stereo is not None and cL and cR:
            Xl = self.state.stereo.triangulate_midpoint((cL[0], cL[1]), (cR[0], cR[1]))  # LEFT frame
            if self.var_show_frame.get() == "zaber":
                # Apply LEFT→ZABER (with optional global scale)
                Xz = (self.state.scale * (self.state.T_left_to_zaber.R @ Xl)) + self.state.T_left_to_zaber.t
                pred_txt = f"Prediction (ZABER): ({Xz[0]:.3f}, {Xz[1]:.3f}, {Xz[2]:.3f}) µm"
            else:
                pred_txt = f"Prediction (LEFT): ({Xl[0]:.3f}, {Xl[1]:.3f}, {Xl[2]:.3f})"
        self.lbl_pred.configure(text=pred_txt)

    def _draw_blank(self, canvas: tk.Canvas):
        canvas.delete("all")
        Wc, Hc = 480, 360
        canvas.create_rectangle(0, 0, Wc, Hc, fill="#111", outline="#111")

    def _draw_img(self, canvas: tk.Canvas, img_bgr, dot_info):
        h, w = img_bgr.shape[:2]
        Hc, Wc = 360, 480
        scale = min(Wc / w, Hc / h)
        vis = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        if dot_info is not None:
            cx, cy, area = dot_info
            x = int(cx * scale); y = int(cy * scale)
            cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
            cv2.putText(vis, f"({cx:.1f},{cy:.1f}) a={area:.0f}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showerror("Error", "Pillow is required (pip install pillow)")
            return
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.image = imgtk
        canvas.delete("all")
        canvas.create_image(Wc // 2, Hc // 2, image=imgtk, anchor="center")


if __name__ == "__main__":
    app = PointCaptureGUI()
    app.geometry("1200x900")
    app.mainloop()
