from __future__ import annotations

import sys
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QShortcut, QTextEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence

# --- your existing pieces ---
from brillouin_system.eye_tracker.stereo_imaging.detect_dot import detect_dot_with_blob  # :contentReference[oaicite:0]{index=0}
from brillouin_system.eye_tracker.stereo_imaging.init_stereo_cameras import stereo_cameras  # :contentReference[oaicite:1]{index=1}
from brillouin_system.eye_tracker.stereo_imaging.fit_coordinate_system import fit_coordinate_system  # :contentReference[oaicite:2]{index=2}
from brillouin_system.eye_tracker.stereo_imaging.se3 import SE3  # :contentReference[oaicite:3]{index=3}

# ---------------- helpers: pixmap ----------------
def numpy_to_qpixmap_gray(arr: np.ndarray) -> QPixmap:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    bytes_per_line = int(arr.strides[0])
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)

def numpy_to_qpixmap_rgb(arr_rgb: np.ndarray) -> QPixmap:
    if arr_rgb.dtype != np.uint8:
        arr_rgb = arr_rgb.astype(np.uint8, copy=False)
    if not arr_rgb.flags.c_contiguous:
        arr_rgb = np.ascontiguousarray(arr_rgb)
    h, w, _ = arr_rgb.shape
    bytes_per_line = int(arr_rgb.strides[0])
    qimg = QImage(arr_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ---------------- helpers: SE3 JSON I/O ----------------
def save_se3_json(T: SE3, path: str | Path) -> None:
    data = {"R": T.R.tolist(), "t": T.t.tolist()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ---------------- data structures ----------------
@dataclass
class PairEntry:
    base: str                 # base name without _left/_right suffix, e.g. prefix_pair_0003
    left_path: Path
    right_path: Path
    coord_path: Optional[Path]  # may be None

    # results (filled later)
    uvL: Optional[Tuple[float, float]] = None
    uvR: Optional[Tuple[float, float]] = None
    Xw: Optional[np.ndarray] = None      # (3,) stereo triangulated in LEFT/world
    rms_px: Optional[float] = None       # reprojection RMS from triangulate_best
    gt: Optional[np.ndarray] = None      # (3,) parsed from coord txt

# ---------------- main GUI ----------------
class StereoFitGUI(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo: Dot-to-World Fitting")
        self.stereo = stereo_cameras  # already built from your JSON calib :contentReference[oaicite:4]{index=4}

        # UI
        self.folder_btn = QPushButton("Choose Folder")
        self.folder_edit = QLineEdit()
        self.scan_btn = QPushButton("Scan & Compute")
        self.fit_btn = QPushButton("Fit Transform & Save")
        self.overlay_check = QCheckBox("Overlay dot & 3D")
        self.overlay_check.setChecked(True)

        self.left_label = QLabel("Left")
        self.right_label = QLabel("Right")
        for lbl in (self.left_label, self.right_label):
            lbl.setFixedSize(640, 480)
            lbl.setStyleSheet("background-color: black;")
            lbl.setAlignment(Qt.AlignCenter)

        self.prev_btn = QPushButton("⟵ Prev")
        self.next_btn = QPushButton("Next ⟶")
        self.idx_label = QLabel("0/0")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        top = QHBoxLayout()
        top.addWidget(self.folder_btn)
        top.addWidget(self.folder_edit)
        top.addWidget(self.scan_btn)
        top.addWidget(self.overlay_check)
        top.addWidget(self.fit_btn)

        imgs = QHBoxLayout()
        imgs.addWidget(self.left_label)
        imgs.addWidget(self.right_label)

        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addStretch()
        nav.addWidget(self.idx_label)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(imgs)
        layout.addLayout(nav)
        layout.addWidget(self.log)
        self.setLayout(layout)

        # state
        self.root: Optional[Path] = None
        self.pairs: List[PairEntry] = []
        self.i = 0

        # wire
        self.folder_btn.clicked.connect(self.on_choose_folder)
        self.scan_btn.clicked.connect(self.on_scan_compute)
        self.fit_btn.clicked.connect(self.on_fit_save)
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)
        QShortcut(QKeySequence("Left"), self, activated=self.on_prev)
        QShortcut(QKeySequence("Right"), self, activated=self.on_next)
        self.status_changed.connect(self._set_status)

    # ------------- UI helpers -------------
    def _set_status(self, s: str):
        self.log.append(s)
        self.log.ensureCursorVisible()

    def _set_pixmap_fit(self, label: QLabel, pm: QPixmap):
        if pm.width() == label.width() and pm.height() == label.height():
            label.setPixmap(pm)
            return
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    # ------------- scan / load -------------
    def on_choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not path:
            return
        self.folder_edit.setText(path)

    def on_scan_compute(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            self.status_changed.emit("Pick a folder first.")
            return
        self.root = Path(folder)
        Ldir = self.root / "left"
        Rdir = self.root / "right"
        Cdir = self.root / "coordinates"
        if not (Ldir.is_dir() and Rdir.is_dir()):
            self.status_changed.emit("Folder must contain 'left/' and 'right/'.")
            return

        # collect pairs by left-filename pattern: replace '_left' → '_right', look for coordinate text
        pairs: List[PairEntry] = []
        left_pngs = sorted(Ldir.glob("*.png"))
        pat = re.compile(r"(.+)_left\.png$", re.IGNORECASE)
        for lp in left_pngs:
            m = pat.match(lp.name)
            if not m:
                continue
            base = m.group(1)
            rp = Rdir / f"{base}_right.png"
            if not rp.exists():
                continue
            cp = (Cdir / f"{base}.txt") if Cdir.is_dir() and (Cdir / f"{base}.txt").exists() else None
            pairs.append(PairEntry(base=base, left_path=lp, right_path=rp, coord_path=cp))
        if not pairs:
            self.status_changed.emit("No matching left/right pairs found.")
            return

        self.pairs = pairs
        self.i = 0
        self.status_changed.emit(f"Found {len(self.pairs)} pairs. Detecting dots and triangulating…")
        self._compute_all()  # detect & triangulate
        self._render()
        self.status_changed.emit("Done.")

    def _compute_all(self):
        for k, p in enumerate(self.pairs):
            # Load grayscale images
            L = cv2.imread(str(p.left_path), cv2.IMREAD_GRAYSCALE)
            R = cv2.imread(str(p.right_path), cv2.IMREAD_GRAYSCALE)
            if L is None or R is None:
                self.status_changed.emit(f"Pair {k}: failed to read images")
                continue

            # Dot detect
            uvL = detect_dot_with_blob(L)    # :contentReference[oaicite:5]{index=5}
            uvR = detect_dot_with_blob(R)
            p.uvL, p.uvR = uvL, uvR

            # Parse ground truth if available
            if p.coord_path is not None:
                try:
                    txt = (p.coord_path.read_text(encoding="utf-8").strip().splitlines()[0]).strip()
                    parts = txt.replace(",", " ").split()
                    if len(parts) >= 3:
                        p.gt = np.array([float(parts[0]), float(parts[1]), float(parts[2])], float)
                except Exception:
                    p.gt = None

            # Triangulate with best method (linear + midpoint + optional LM refine) :contentReference[oaicite:6]{index=6}
            if uvL is not None and uvR is not None:
                try:
                    Xw, rms = self.stereo.triangulate_best(uvL, uvR, refine=True)  # best available
                    p.Xw = Xw
                    p.rms_px = rms
                except Exception as e:
                    self.status_changed.emit(f"{p.base}: triangulation error: {e}")

    # ------------- render -------------
    def _render(self):
        if not self.pairs:
            self.left_label.setText("Left")
            self.right_label.setText("Right")
            self.idx_label.setText("0/0")
            return

        n = len(self.pairs)
        i = max(0, min(self.i, n - 1))
        self.i = i
        self.idx_label.setText(f"{i+1}/{n}")

        p = self.pairs[i]
        L = cv2.imread(str(p.left_path), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(p.right_path), cv2.IMREAD_GRAYSCALE)

        def overlay(gray, uv, Xw):
            if gray is None:
                return None
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if uv is not None and self.overlay_check.isChecked():
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.drawMarker(vis, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
                cv2.circle(vis, (u, v), 6, (0, 255, 0), 2, cv2.LINE_AA)
            if Xw is not None and self.overlay_check.isChecked():
                txt = f"X=({Xw[0]:.2f}, {Xw[1]:.2f}, {Xw[2]:.2f})"
                cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            return vis

        Lvis = overlay(L, p.uvL, p.Xw)
        Rvis = overlay(R, p.uvR, p.Xw)

        self._set_pixmap_fit(self.left_label, numpy_to_qpixmap_rgb(Lvis) if Lvis is not None else numpy_to_qpixmap_gray(L))
        self._set_pixmap_fit(self.right_label, numpy_to_qpixmap_rgb(Rvis) if Rvis is not None else numpy_to_qpixmap_gray(R))

        # log line
        gt_str = "none" if p.gt is None else f"({p.gt[0]:.3f},{p.gt[1]:.3f},{p.gt[2]:.3f})"
        tri_str = "none" if p.Xw is None else f"({p.Xw[0]:.3f},{p.Xw[1]:.3f},{p.Xw[2]:.3f})"
        err_str = ""
        if p.gt is not None and p.Xw is not None:
            err = np.linalg.norm(p.Xw - p.gt)
            err_str = f" | ‖err‖={err:.3f}"
        self.status_changed.emit(f"{p.base}: uvL={p.uvL} uvR={p.uvR} | tri={tri_str} | gt={gt_str}{err_str} | rms={p.rms_px if p.rms_px is not None else 'n/a'}px")

    # ------------- nav -------------
    def on_prev(self):
        if not self.pairs: return
        self.i = max(0, self.i - 1)
        self._render()

    def on_next(self):
        if not self.pairs: return
        self.i = min(len(self.pairs) - 1, self.i + 1)
        self._render()

    # ------------- fit & save -------------
    def on_fit_save(self):
        if not self.pairs:
            self.status_changed.emit("Nothing to fit.")
            return

        # Collect valid correspondences (stereo result ↔ stored gt)
        A_left = []
        B_gt = []
        used = []
        for p in self.pairs:
            if p.Xw is not None and p.gt is not None and np.isfinite(p.Xw).all() and np.isfinite(p.gt).all():
                A_left.append(p.Xw.reshape(1, 3))
                B_gt.append(p.gt.reshape(1, 3))
                used.append(p.base)

        if len(A_left) < 3:
            self.status_changed.emit("Need at least 3 valid pairs with both stereo and GT coordinates.")
            return

        A = np.vstack(A_left)
        B = np.vstack(B_gt)

        # Optional: trimming and (no-scale) rigid fit are sensible defaults
        with_scale = False
        trim_fraction = 0.2   # keep best 80%
        T_left_to_gt, info = fit_coordinate_system(
            points_left=A, points_zaber=B,
            with_scale=with_scale, trim_fraction=trim_fraction, trim_repeats=2
        )  # :contentReference[oaicite:7]{index=7}

        self.status_changed.emit(
            f"Fit done: rms={info['rms']:.4f}, inliers={info['num_inliers']}/{len(A)}, scale={info['scale']:.6f}"
        )

        # Save JSON
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Transform JSON", "T_left_to_stage.json", "JSON (*.json)")
        if out_path:
            save_se3_json(T_left_to_gt, out_path)
            self.status_changed.emit(f"Saved transform to: {out_path}")

# ---------------- main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = StereoFitGUI()
    w.resize(1100, 820)
    w.show()
    sys.exit(app.exec_())
