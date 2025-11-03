#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QCheckBox, QComboBox
)

# Matplotlib (Qt5Agg)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_Rt(obj):
    R = np.asarray(obj["R"], float).reshape(3, 3)
    t = np.asarray(obj["t"], float).reshape(3)
    return R, t


class CalibViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transform & Points Viewer")

        # --- UI
        self.load_btn = QPushButton("Load Transform JSON")
        self.status = QLabel("Load a JSON to begin")
        self.status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.show_inliers_chk = QCheckBox("Highlight inliers")
        self.show_inliers_chk.setChecked(True)

        self.points_source_combo = QComboBox()
        self.points_source_combo.addItems([
            "All pairs (raw)",
            "Valid pairs (used for fit)",
        ])

        header = QHBoxLayout()
        header.addWidget(self.load_btn)
        header.addStretch()
        header.addWidget(QLabel("Show:"))
        header.addWidget(self.points_source_combo)
        header.addWidget(self.show_inliers_chk)

        # Matplotlib figure
        self.fig = Figure(figsize=(7, 6), constrained_layout=True)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout()
        layout.addLayout(header)
        layout.addWidget(self.canvas)
        layout.addWidget(self.status)
        self.setLayout(layout)

        # data slots
        self.R = None
        self.t = None
        self.data = None
        self.source_info = {}

        # wire
        self.load_btn.clicked.connect(self.on_load)
        self.points_source_combo.currentIndexChanged.connect(self.redraw)
        self.show_inliers_chk.toggled.connect(self.redraw)

        self._init_axes()

    def _init_axes(self):
        self.ax.clear()
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Coordinate Frames & Points")
        self.canvas.draw_idle()

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Transform JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            self.status.setText(f"Failed to load: {e}")
            return

        try:
            self.R, self.t = _as_Rt(obj)
        except Exception as e:
            self.status.setText(f"Invalid JSON (R,t): {e}")
            return

        self.data = _safe_get(obj, "data", default={}) or {}
        self.source_info = _safe_get(obj, "source", default={}) or {}

        # Basic stats
        rms = _safe_get(self.source_info, "rms_error", default=float("nan"))
        n = _safe_get(self.source_info, "num_points", default=None)
        tgt = _safe_get(self.source_info, "target_frame", default="")
        name = _safe_get(obj, "name", default=Path(path).stem)

        self.status.setText(
            f"Loaded '{name}'  |  target='{tgt}'  |  N={n}  |  RMS={rms:.4f}"
            if isinstance(rms, (int, float))
            else f"Loaded '{name}'  |  target='{tgt}'  |  N={n}"
        )

        self.redraw()

    # ------------- plotting helpers -------------
    def _plot_axes(self, origin, R, length=10.0, alpha=1.0, lw=2.0, label_prefix=""):
        """
        Plot coordinate axes defined by rotation R at origin.
        """
        o = origin.reshape(3)
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]  # columns are axis directions
        X = np.vstack([o, o + length * ex])
        Y = np.vstack([o, o + length * ey])
        Z = np.vstack([o, o + length * ez])

        # Don't set explicit colors (let mpl choose) unless you want RGB:
        self.ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=lw, alpha=alpha, label=f"{label_prefix}X")
        self.ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], linewidth=lw, alpha=alpha, label=f"{label_prefix}Y")
        self.ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], linewidth=lw, alpha=alpha, label=f"{label_prefix}Z")

        # axis tips markers
        self.ax.scatter([X[-1, 0]], [X[-1, 1]], [X[-1, 2]], s=20, alpha=alpha)
        self.ax.scatter([Y[-1, 0]], [Y[-1, 1]], [Y[-1, 2]], s=20, alpha=alpha)
        self.ax.scatter([Z[-1, 0]], [Z[-1, 1]], [Z[-1, 2]], s=20, alpha=alpha)

    def _extract_pairs(self):
        """
        Returns (A, B, mask_inliers, allA, allB)
        A,B are arrays for the selected source (“All pairs” or “Valid pairs”).
        mask_inliers is aligned with A,B if available; otherwise None.
        allA,allB are the raw arrays for reference (None if not available).
        """
        all_pairs = self.data.get("all_pairs", None)
        valid_pairs = self.data.get("valid_pairs", None)
        inliers = self.data.get("inliers", None)

        allA = allB = None
        if isinstance(all_pairs, list):
            # convert to arrays (None rows become NaNs)
            A_list, B_list = [], []
            for item in all_pairs:
                l = item.get("left", None)
                r = item.get("input", None)
                A_list.append([np.nan, np.nan, np.nan] if l is None else list(map(float, l)))
                B_list.append([np.nan, np.nan, np.nan] if r is None else list(map(float, r)))
            allA = np.asarray(A_list, float)
            allB = np.asarray(B_list, float)

        if self.points_source_combo.currentIndex() == 0:
            # All pairs (raw)
            A = allA
            B = allB
            mask = None  # no inlier mask at "raw" level
        else:
            # Valid pairs (used for fit)
            A = np.asarray(_safe_get(valid_pairs, "left", default=[]), float)
            B = np.asarray(_safe_get(valid_pairs, "input", default=[]), float)
            mask = _safe_get(inliers, "mask", default=None)
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape[0] != A.shape[0]:
                    mask = None  # safety

        return A, B, mask, allA, allB

    def _set_equal_3d(self, X):
        """
        Set equal aspect box around data X (Nx3).
        """
        if X is None or X.size == 0 or not np.isfinite(X).any():
            self.ax.set_box_aspect((1, 1, 1))
            return
        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            self.ax.set_box_aspect((1, 1, 1))
            return
        spans = np.maximum(maxs - mins, 1e-9)
        center = (maxs + mins) / 2.0
        half = np.max(spans) / 2.0
        lims = np.vstack([center - half, center + half])
        self.ax.set_xlim(lims[:, 0])
        self.ax.set_ylim(lims[:, 1])
        self.ax.set_zlim(lims[:, 2])
        self.ax.set_box_aspect((1, 1, 1))

    # ------------- main redraw -------------
    def redraw(self):
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Coordinate Frames & Points")

        # Nothing loaded yet
        if self.R is None or self.t is None:
            self.canvas.draw_idle()
            return

        R = self.R
        t = self.t

        # Frames:
        # LEFT frame at origin (I, 0)
        self._plot_axes(np.zeros(3), np.eye(3), length=10.0, alpha=0.7, lw=1.5, label_prefix="L-")

        # TARGET frame at (t) with orientation R
        self._plot_axes(t, R, length=10.0, alpha=1.0, lw=2.5, label_prefix="T-")

        # Points
        A, B, mask, allA, allB = self._extract_pairs()

        scatter_sets = []

        if A is not None and A.size:
            # LEFT points (used source)
            finiteA = np.isfinite(A).all(axis=1)
            self.ax.scatter(A[finiteA, 0], A[finiteA, 1], A[finiteA, 2], marker="o", s=20, label="LEFT pts")
            scatter_sets.append(A[finiteA])

        if B is not None and B.size:
            # target points (used source)
            finiteB = np.isfinite(B).all(axis=1)
            self.ax.scatter(B[finiteB, 0], B[finiteB, 1], B[finiteB, 2], marker="^", s=24, label="TARGET pts")
            scatter_sets.append(B[finiteB])

        # Inliers highlighting (only for "valid pairs")
        if self.points_source_combo.currentIndex() == 1 and mask is not None and self.show_inliers_chk.isChecked():
            # emphasize inliers by circling them
            idxs = np.where(mask & np.isfinite(A).all(axis=1))[0]
            for i in idxs:
                self.ax.scatter([A[i, 0]], [A[i, 1]], [A[i, 2]], s=80, facecolors="none", edgecolors="k", linewidths=1.0)

        # Draw lines between matched pairs (valid view only)
        if self.points_source_combo.currentIndex() == 1 and A is not None and B is not None and A.shape == B.shape:
            finite = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
            for i in np.where(finite)[0]:
                seg = np.vstack([A[i], B[i]])
                self.ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=0.8, alpha=0.6)

        # Fit bounding box
        if scatter_sets:
            X = np.vstack(scatter_sets) if len(scatter_sets) > 1 else scatter_sets[0]
            self._set_equal_3d(X)

        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = CalibViewer()
    w.resize(900, 750)
    w.show()
    sys.exit(app.exec_())
