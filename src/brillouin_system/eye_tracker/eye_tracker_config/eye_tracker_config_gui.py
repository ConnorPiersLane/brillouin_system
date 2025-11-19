# eye_tracker_config_gui.py (updated)

from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
    QApplication, QMessageBox, QCheckBox, QComboBox, QFileDialog
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt

from dataclasses import asdict
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

        self.inputs = {}
        layout = QVBoxLayout()
        layout.addWidget(self._group_main(self.inputs))
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self._load()

    def _group_main(self, inputs):
        g = QGroupBox("Settings")
        v = QVBoxLayout()

        def add_row(label, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget, 1)
            v.addLayout(h)

        # Sleep
        w = QCheckBox("Sleep / Pause processing")
        inputs["sleep"] = w
        add_row("Sleep", w)

        # Thresholds
        le = QLineEdit(); le.setValidator(QIntValidator(0, 255)); inputs["binary_threshold_left"] = le
        add_row("Binary threshold (left)", le)
        re = QLineEdit(); re.setValidator(QIntValidator(0, 255)); inputs["binary_threshold_right"] = re
        add_row("Binary threshold (right)", re)

        # Save images options
        chk_save = QCheckBox("Enable saving")
        inputs["save_images"] = chk_save
        add_row("Save images", chk_save)

        spath = QLineEdit(); inputs["save_images_path"] = spath
        btn_browse = QPushButton("Browse…")
        def browse():
            d = QFileDialog.getExistingDirectory(self, "Select folder", "")
            if d: spath.setText(d)
        btn_browse.clicked.connect(browse)
        h = QHBoxLayout(); h.addWidget(spath, 1); h.addWidget(btn_browse)
        hv = QVBoxLayout(); hv.addWidget(QLabel("Save folder")); hv.addLayout(h)
        v.addLayout(hv)

        freq = QLineEdit(); freq.setValidator(QIntValidator(1, 1000)); inputs["max_saving_freq_hz"] = freq
        add_row("Max saving frequency (Hz)", freq)

        # Ellipse fitting / overlay
        de = QCheckBox("Run ellipse fitting"); inputs["do_ellipse_fitting"] = de
        add_row("Ellipse fitting", de)
        ov = QCheckBox("Overlay ellipse on output"); inputs["overlay_ellipse"] = ov
        add_row("Overlay ellipse", ov)

        # Frame returned
        cb = QComboBox()
        cb.addItems(["original", "binary", "floodfilled", "contour"])
        inputs["frame_returned"] = cb
        add_row("Frame returned", cb)

        g.setLayout(v)
        return g

    def _buttons(self):
        h = QHBoxLayout()
        apply_btn = QPushButton("Apply"); apply_btn.clicked.connect(self.apply)
        save_btn = QPushButton("Save"); save_btn.clicked.connect(self.save)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.close)
        h.addStretch(); h.addWidget(apply_btn); h.addWidget(save_btn); h.addWidget(close_btn)
        return h

    def _set_fields(self, cfg: EyeTrackerConfig):
        self.inputs["sleep"].setChecked(bool(cfg.sleep))
        self.inputs["binary_threshold_left"].setText(str(cfg.binary_threshold_left))
        self.inputs["binary_threshold_right"].setText(str(cfg.binary_threshold_right))
        self.inputs["save_images"].setChecked(bool(cfg.save_images))
        self.inputs["save_images_path"].setText(cfg.save_images_path or "")
        self.inputs["max_saving_freq_hz"].setText(str(cfg.max_saving_freq_hz))
        self.inputs["do_ellipse_fitting"].setChecked(bool(cfg.do_ellipse_fitting))
        self.inputs["overlay_ellipse"].setChecked(bool(cfg.overlay_ellipse))
        idx = ["original","binary","floodfilled"].index(cfg.frame_returned if cfg.frame_returned in ("original","binary","floodfilled") else "original")
        self.inputs["frame_returned"].setCurrentIndex(idx)

    def _collect(self) -> dict:
        return {
            "sleep": bool(self.inputs["sleep"].isChecked()),
            "binary_threshold_left": int(self.inputs["binary_threshold_left"].text() or 20),
            "binary_threshold_right": int(self.inputs["binary_threshold_right"].text() or 20),
            "save_images": bool(self.inputs["save_images"].isChecked()),
            "save_images_path": self.inputs["save_images_path"].text(),
            "max_saving_freq_hz": int(self.inputs["max_saving_freq_hz"].text() or 5),
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
