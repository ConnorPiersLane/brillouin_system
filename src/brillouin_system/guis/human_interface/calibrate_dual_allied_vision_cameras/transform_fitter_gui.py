#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform Fitter GUI
- Load a coordinates JSON file (entries with 'left_uv' and 'inserted_xyz'), produced by the capture GUI.
- Fit a rigid (or similarity) transform between LEFT 3D points and target coordinates.
- Save the resulting transform in a JSON (keys: R, t, plus some metadata).
"""

import sys
import json
import time
from typing import List, Tuple, Dict

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
    QFormLayout, QLineEdit
)

# ----- Project-local utilities (provided/uploaded by you) -----
# SE3 tiny class + (de)serialization helpers
from se3 import SE3  # local file se3.py
# Fitting routine (Umeyama/Kabsch) with optional trimming and scale
from fit_coordinate_system import fit_coordinate_system  # local file fit_coordinate_system.py


def _load_capture_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "entries" not in d or not isinstance(d["entries"], list):
        raise ValueError("JSON missing 'entries' list")
    return d["entries"]


def _extract_points(entries: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (left_points_3d, target_points_3d).
    The capture JSON stores 'left_uv' (2D) and 'inserted_xyz' (3D). For fitting a 3D-3D transform,
    we assume the inserted_xyz are the ground-truth in the target frame.
    For LEFT 3D: the capture pipeline can store triangulated 3D in future, but here we assume
    the saved XYZ are already in the target frame. Therefore, this GUI expects a JSON that
    already contains left-frame 3D. If your saved JSON only has (u,v) and (x,y,z), provide
    your own loader that maps (u,v) to 3D before calling this GUI.
    """
    # For this minimal GUI, we expect fields named 'left_xyz' and 'inserted_xyz'.
    left_xyz = []
    tgt_xyz = []
    for e in entries:
        if "left_xyz" in e and "inserted_xyz" in e:
            lx = np.asarray(e["left_xyz"], float).reshape(3)
            tx = np.asarray(e["inserted_xyz"], float).reshape(3)
            left_xyz.append(lx)
            tgt_xyz.append(tx)
    if not left_xyz:
        raise ValueError("No 3D correspondences found. Expected 'left_xyz' and 'inserted_xyz' in entries.")
    return np.vstack(left_xyz), np.vstack(tgt_xyz)


def _save_transform_json(path: str, T: SE3, meta: Dict):
    payload = {
        "name": meta.get("name", "left_to_target"),
        "description": meta.get("description", "Transformation from left camera frame to target coordinate system"),
        "R": T.R.tolist(),
        "t": T.t.tolist(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "method": "umeyama_fit",
            "num_points": int(meta.get("num_points", 0)),
            "rms_error": float(meta.get("rms_error", float("nan"))),
            "scale": float(meta.get("scale", 1.0)),
            "trim_fraction": float(meta.get("trim_fraction", 0.0)),
            "trim_repeats": int(meta.get("trim_repeats", 0)),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class FitterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transform Fitter (LEFT → Target)")
        self.resize(560, 360)

        self.loaded_path = None
        self.entries = None
        self.left_pts = None
        self.tgt_pts = None
        self.result_T = None
        self.result_info = None

        # --- File controls ---
        self.load_btn = QPushButton("Load Coordinates JSON…")
        self.save_btn = QPushButton("Save Transform JSON…"); self.save_btn.setEnabled(False)

        self.loaded_label = QLineEdit(); self.loaded_label.setReadOnly(True)

        file_row = QHBoxLayout()
        file_row.addWidget(self.load_btn)
        file_row.addWidget(self.save_btn)

        # --- Fit options ---
        opt_box = QGroupBox("Fit Options")
        form = QFormLayout()

        self.with_scale = QCheckBox("Estimate global scale (similarity)")
        self.trim_frac = QDoubleSpinBox(); self.trim_frac.setRange(0.0, 0.49); self.trim_frac.setSingleStep(0.05); self.trim_frac.setValue(0.0)
        self.trim_rep = QSpinBox(); self.trim_rep.setRange(1, 10); self.trim_rep.setValue(1)

        self.name_edit = QLineEdit("left_to_target")
        self.desc_edit = QLineEdit("Transformation from left camera to target frame")

        form.addRow(self.with_scale)
        form.addRow("Trim fraction:", self.trim_frac)
        form.addRow("Trim repeats:", self.trim_rep)
        form.addRow("Output name:", self.name_edit)
        form.addRow("Output description:", self.desc_edit)
        opt_box.setLayout(form)

        # --- Actions / status ---
        self.fit_btn = QPushButton("Fit Transform"); self.fit_btn.setEnabled(False)

        self.npts_label = QLabel("Points: –")
        self.inliers_label = QLabel("Inliers: –")
        self.rms_label = QLabel("RMS: –")
        self.scale_label = QLabel("Scale: –")
        self.R_label = QLabel("R: –")
        self.t_label = QLabel("t: –")

        stats = QVBoxLayout()
        stats.addWidget(self.npts_label)
        stats.addWidget(self.inliers_label)
        stats.addWidget(self.rms_label)
        stats.addWidget(self.scale_label)
        stats.addWidget(self.R_label)
        stats.addWidget(self.t_label)

        # --- Layout ---
        lay = QVBoxLayout()
        lay.addLayout(file_row)
        lay.addWidget(self.loaded_label)
        lay.addWidget(opt_box)

        bottom = QHBoxLayout()
        bottom.addWidget(self.fit_btn)
        bottom.addStretch()
        bottom.addLayout(stats)

        lay.addLayout(bottom)
        self.setLayout(lay)

        # --- Signals ---
        self.load_btn.clicked.connect(self.on_load_json)
        self.fit_btn.clicked.connect(self.on_fit)
        self.save_btn.clicked.connect(self.on_save_json)

    def on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Coordinates JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            entries = _load_capture_json(path)
            # Extract 3D-3D correspondences
            L, T = _extract_points(entries)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Could not load 3D correspondences:\n{e}")
            return
        self.loaded_path = path
        self.entries = entries
        self.left_pts = L
        self.tgt_pts = T
        self.loaded_label.setText(path)
        self.npts_label.setText(f"Points: {L.shape[0]}")
        self.fit_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.result_T = None
        self.result_info = None
        self._update_stats(None, None)

    def _update_stats(self, T: SE3 | None, info: dict | None):
        if T is None or info is None:
            self.inliers_label.setText("Inliers: –")
            self.rms_label.setText("RMS: –")
            self.scale_label.setText("Scale: –")
            self.R_label.setText("R: –")
            self.t_label.setText("t: –")
            return
        self.inliers_label.setText(f"Inliers: {info.get('num_inliers', '–')}")
        self.rms_label.setText(f"RMS: {info.get('rms', float('nan')):.4f}")
        self.scale_label.setText(f"Scale: {info.get('scale', 1.0):.6f}")
        R = np.asarray(T.R)
        t = np.asarray(T.t)
        self.R_label.setText("R:\n" + np.array2string(R, precision=5, suppress_small=True))
        self.t_label.setText("t:\n" + np.array2string(t, precision=5, suppress_small=True))

    def on_fit(self):
        if self.left_pts is None or self.tgt_pts is None:
            return
        try:
            T, info = fit_coordinate_system(
                self.left_pts, self.tgt_pts,
                with_scale=self.with_scale.isChecked(),
                trim_fraction=self.trim_frac.value(),
                trim_repeats=self.trim_rep.value()
            )
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", f"Fitting failed:\n{e}")
            return
        self.result_T = T
        self.result_info = info
        self._update_stats(T, info)
        self.save_btn.setEnabled(True)

    def on_save_json(self):
        if self.result_T is None or self.result_info is None:
            QMessageBox.information(self, "Nothing to save", "Run the fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Transform JSON", "", "JSON (*.json)")
        if not path:
            return
        meta = {
            "name": self.name_edit.text().strip() or "left_to_target",
            "description": self.desc_edit.text().strip() or "Transformation from left camera to target frame",
            "num_points": int(self.left_pts.shape[0]) if self.left_pts is not None else 0,
            "rms_error": float(self.result_info.get("rms", float("nan"))),
            "scale": float(self.result_info.get("scale", 1.0)),
            "trim_fraction": float(self.trim_frac.value()),
            "trim_repeats": int(self.trim_rep.value()),
        }
        try:
            _save_transform_json(path, self.result_T, meta)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save transform:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Saved transform JSON to:\n{path}")


def main():
    app = QApplication(sys.argv)
    w = FitterGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
