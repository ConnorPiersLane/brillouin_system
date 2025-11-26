# eye_tracker_config_gui.py

from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
    QApplication, QMessageBox, QCheckBox, QComboBox, QFileDialog
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt

from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import (
    EyeTrackerConfig, PUPIL_FIT_TOML_PATH,
    load_eye_tracker_config, save_config_section,
)
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig


class EyeTrackerConfigDialog(QDialog):
    def __init__(self, cfg_holder: ThreadSafeConfig, on_apply=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eye Tracker Configuration")
        self.cfg_holder = cfg_holder
        self.on_apply = on_apply

        self.inputs: dict[str, object] = {}
        layout = QVBoxLayout()
        layout.addWidget(self._group_main(self.inputs))
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self._load()

    def _group_main(self, inputs: dict) -> QGroupBox:
        g = QGroupBox("Settings")
        v = QVBoxLayout()

        def add_row(label: str, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget, 1)
            v.addLayout(h)

        # Sleep
        w_sleep = QCheckBox("Sleep / Pause processing")
        inputs["sleep"] = w_sleep
        add_row("Sleep", w_sleep)

        # Thresholds
        le_left_thr = QLineEdit()
        le_left_thr.setValidator(QIntValidator(0, 255))
        inputs["binary_threshold_left"] = le_left_thr
        add_row("Binary threshold (left)", le_left_thr)

        le_right_thr = QLineEdit()
        le_right_thr.setValidator(QIntValidator(0, 255))
        inputs["binary_threshold_right"] = le_right_thr
        add_row("Binary threshold (right)", le_right_thr)

        # Per-eye ROIs
        # Left eye ROI
        left_roi_group = QGroupBox("Left eye ROI")
        left_roi_layout = QVBoxLayout()

        left_center_layout = QHBoxLayout()
        left_center_layout.addWidget(QLabel("Center X"))
        le_lcx = QLineEdit()
        le_lcx.setValidator(QIntValidator(0, 10000))
        inputs["roi_left_center_x"] = le_lcx
        left_center_layout.addWidget(le_lcx)

        left_center_layout.addWidget(QLabel("Center Y"))
        le_lcy = QLineEdit()
        le_lcy.setValidator(QIntValidator(0, 10000))
        inputs["roi_left_center_y"] = le_lcy
        left_center_layout.addWidget(le_lcy)
        left_roi_layout.addLayout(left_center_layout)

        left_size_layout = QHBoxLayout()
        left_size_layout.addWidget(QLabel("Width"))
        le_lw = QLineEdit()
        le_lw.setValidator(QIntValidator(1, 10000))
        inputs["roi_left_width"] = le_lw
        left_size_layout.addWidget(le_lw)

        left_size_layout.addWidget(QLabel("Height"))
        le_lh = QLineEdit()
        le_lh.setValidator(QIntValidator(1, 10000))
        inputs["roi_left_height"] = le_lh
        left_size_layout.addWidget(le_lh)
        left_roi_layout.addLayout(left_size_layout)

        left_roi_group.setLayout(left_roi_layout)
        v.addWidget(left_roi_group)

        # Right eye ROI
        right_roi_group = QGroupBox("Right eye ROI")
        right_roi_layout = QVBoxLayout()

        right_center_layout = QHBoxLayout()
        right_center_layout.addWidget(QLabel("Center X"))
        le_rcx = QLineEdit()
        le_rcx.setValidator(QIntValidator(0, 10000))
        inputs["roi_right_center_x"] = le_rcx
        right_center_layout.addWidget(le_rcx)

        right_center_layout.addWidget(QLabel("Center Y"))
        le_rcy = QLineEdit()
        le_rcy.setValidator(QIntValidator(0, 10000))
        inputs["roi_right_center_y"] = le_rcy
        right_center_layout.addWidget(le_rcy)
        right_roi_layout.addLayout(right_center_layout)

        right_size_layout = QHBoxLayout()
        right_size_layout.addWidget(QLabel("Width"))
        le_rw = QLineEdit()
        le_rw.setValidator(QIntValidator(1, 10000))
        inputs["roi_right_width"] = le_rw
        right_size_layout.addWidget(le_rw)

        right_size_layout.addWidget(QLabel("Height"))
        le_rh = QLineEdit()
        le_rh.setValidator(QIntValidator(1, 10000))
        inputs["roi_right_height"] = le_rh
        right_size_layout.addWidget(le_rh)
        right_roi_layout.addLayout(right_size_layout)

        right_roi_group.setLayout(right_roi_layout)
        v.addWidget(right_roi_group)

        # Show ROI overlay
        chk_apply_roi = QCheckBox("Draw ROI rectangles on preview")
        inputs["apply_roi"] = chk_apply_roi
        add_row("Show ROI overlay", chk_apply_roi)

        # Save images options
        chk_save = QCheckBox("Enable saving")
        inputs["save_images"] = chk_save
        add_row("Save images", chk_save)

        spath = QLineEdit()
        inputs["save_images_path"] = spath
        btn_browse = QPushButton("Browse…")

        def browse():
            d = QFileDialog.getExistingDirectory(self, "Select folder", "")
            if d:
                spath.setText(d)

        btn_browse.clicked.connect(browse)
        h_path = QHBoxLayout()
        h_path.addWidget(spath, 1)
        h_path.addWidget(btn_browse)
        hv_path = QVBoxLayout()
        hv_path.addWidget(QLabel("Save folder"))
        hv_path.addLayout(h_path)
        v.addLayout(hv_path)

        freq = QLineEdit()
        freq.setValidator(QIntValidator(1, 1000))
        inputs["max_saving_freq_hz"] = freq
        add_row("Max saving frequency (Hz)", freq)

        # Ellipse fitting / overlay
        de = QCheckBox("Run ellipse fitting")
        inputs["do_ellipse_fitting"] = de
        add_row("Ellipse fitting", de)

        ov = QCheckBox("Overlay ellipse on output")
        inputs["overlay_ellipse"] = ov
        add_row("Overlay ellipse", ov)

        # Frame returned
        cb = QComboBox()
        frame_options = ["original", "binary", "floodfilled", "contour"]
        cb.addItems(frame_options)
        inputs["frame_returned"] = cb
        add_row("Frame returned", cb)

        g.setLayout(v)
        return g

    def _buttons(self) -> QHBoxLayout:
        h = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        h.addStretch()
        h.addWidget(apply_btn)
        h.addWidget(save_btn)
        h.addWidget(close_btn)
        return h

    def _set_fields(self, cfg: EyeTrackerConfig):
        # Sleep
        self.inputs["sleep"].setChecked(bool(cfg.sleep))

        # Thresholds
        self.inputs["binary_threshold_left"].setText(str(cfg.binary_threshold_left))
        self.inputs["binary_threshold_right"].setText(str(cfg.binary_threshold_right))

        # ROIs
        lc_x, lc_y = cfg.roi_left_center_xy
        lw, lh = cfg.roi_left_width_height
        rc_x, rc_y = cfg.roi_right_center_xy
        rw, rh = cfg.roi_right_width_height

        self.inputs["roi_left_center_x"].setText(str(lc_x))
        self.inputs["roi_left_center_y"].setText(str(lc_y))
        self.inputs["roi_left_width"].setText(str(lw))
        self.inputs["roi_left_height"].setText(str(lh))

        self.inputs["roi_right_center_x"].setText(str(rc_x))
        self.inputs["roi_right_center_y"].setText(str(rc_y))
        self.inputs["roi_right_width"].setText(str(rw))
        self.inputs["roi_right_height"].setText(str(rh))

        self.inputs["apply_roi"].setChecked(bool(cfg.apply_roi))

        # Saving
        self.inputs["save_images"].setChecked(bool(cfg.save_images))
        self.inputs["save_images_path"].setText(cfg.save_images_path or "")
        self.inputs["max_saving_freq_hz"].setText(str(cfg.max_saving_freq_hz))

        # Ellipse fitting
        self.inputs["do_ellipse_fitting"].setChecked(bool(cfg.do_ellipse_fitting))
        self.inputs["overlay_ellipse"].setChecked(bool(cfg.overlay_ellipse))

        # Frame returned
        frame_options = ["original", "binary", "floodfilled", "contour"]
        try:
            idx = frame_options.index(cfg.frame_returned)
        except ValueError:
            idx = 0
        self.inputs["frame_returned"].setCurrentIndex(idx)

    def _collect(self) -> dict:
        # Helper to read an int with fallback
        def _intval(key: str, default: int) -> int:
            text = self.inputs[key].text()
            return int(text) if text else default

        return {
            "sleep": bool(self.inputs["sleep"].isChecked()),
            "binary_threshold_left": _intval("binary_threshold_left", 20),
            "binary_threshold_right": _intval("binary_threshold_right", 20),

            "roi_left_center_xy": (
                _intval("roi_left_center_x", 500),
                _intval("roi_left_center_y", 500),
            ),
            "roi_left_width_height": (
                _intval("roi_left_width", 100),
                _intval("roi_left_height", 100),
            ),
            "roi_right_center_xy": (
                _intval("roi_right_center_x", 500),
                _intval("roi_right_center_y", 500),
            ),
            "roi_right_width_height": (
                _intval("roi_right_width", 100),
                _intval("roi_right_height", 100),
            ),
            "apply_roi": bool(self.inputs["apply_roi"].isChecked()),

            "save_images": bool(self.inputs["save_images"].isChecked()),
            "save_images_path": self.inputs["save_images_path"].text(),
            "max_saving_freq_hz": _intval("max_saving_freq_hz", 5),

            "do_ellipse_fitting": bool(self.inputs["do_ellipse_fitting"].isChecked()),
            "overlay_ellipse": bool(self.inputs["overlay_ellipse"].isChecked()),
            "frame_returned": self.inputs["frame_returned"].currentText(),
        }

    def _load(self):
        cfg = self.cfg_holder.get()
        self._set_fields(cfg)

    def apply(self):
        try:
            kwargs = self._collect()
            self.cfg_holder.update(**kwargs)
            if self.on_apply:
                self.on_apply(self.cfg_holder.get())
            QMessageBox.information(self, "Applied", "Settings applied (not saved).")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply config:\n{e}")

    def save(self):
        try:
            self.apply()
            save_config_section(PUPIL_FIT_TOML_PATH, "eye_tracker", self.cfg_holder)
            QMessageBox.information(self, "Saved", f"Saved to {PUPIL_FIT_TOML_PATH.name}.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")


if __name__ == "__main__":
    import sys
    # Load existing config and wrap in ThreadSafeConfig
    holder = ThreadSafeConfig(load_eye_tracker_config(PUPIL_FIT_TOML_PATH, "eye_tracker"))
    app = QApplication(sys.argv)
    dlg = EyeTrackerConfigDialog(holder)
    dlg.show()
    sys.exit(app.exec_())
