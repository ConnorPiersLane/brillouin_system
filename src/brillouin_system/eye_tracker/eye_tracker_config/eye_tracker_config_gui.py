# eye_tracker_config_gui.py

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QApplication,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt

from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import (
    EyeTrackerConfig,
    PUPIL_FIT_TOML_PATH,
    load_eye_tracker_config,
    save_config_section,
    eye_tracker_config,
)
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig


class EyeTrackerConfigDialog(QDialog):
    def __init__(
        self,
        cfg_holder: ThreadSafeConfig | None = None,
        on_apply=None,
        parent=None,
    ):
        """
        cfg_holder:
            ThreadSafeConfig[EyeTrackerConfig]. If None, the global
            `eye_tracker_config` is used.

        on_apply:
            Optional callback taking a single EyeTrackerConfig, called
            after cfg_holder has been updated (but before saving).
        """
        super().__init__(parent)
        self.setWindowTitle("Eye Tracker Configuration")

        # Store the ThreadSafeConfig, not the raw dataclass
        self.cfg_holder: ThreadSafeConfig = cfg_holder or eye_tracker_config
        self.on_apply = on_apply

        self.inputs: dict[str, object] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._group_main(self.inputs))
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self._load()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _group_main(self, inputs: dict) -> QGroupBox:
        g = QGroupBox("Settings")
        v = QVBoxLayout()

        def add_row(label: str, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget, 1)
            v.addLayout(h)

        # Sleep / pause
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

    # ------------------------------------------------------------------ #
    # Data <-> UI
    # ------------------------------------------------------------------ #

    def _set_fields(self, cfg: EyeTrackerConfig):
        # Sleep
        self.inputs["sleep"].setChecked(bool(cfg.sleep))

        # Thresholds
        self.inputs["binary_threshold_left"].setText(str(cfg.binary_threshold_left))
        self.inputs["binary_threshold_right"].setText(str(cfg.binary_threshold_right))

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
            "save_images": bool(self.inputs["save_images"].isChecked()),
            "save_images_path": self.inputs["save_images_path"].text(),
            "max_saving_freq_hz": _intval("max_saving_freq_hz", 5),
            "do_ellipse_fitting": bool(self.inputs["do_ellipse_fitting"].isChecked()),
            "overlay_ellipse": bool(self.inputs["overlay_ellipse"].isChecked()),
            "frame_returned": self.inputs["frame_returned"].currentText(),
        }

    def _load(self):
        """Load current config from the ThreadSafeConfig into the widgets."""
        cfg = self.cfg_holder.get()
        self._set_fields(cfg)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def apply(self):
        """Apply settings to in-memory config (but do not save TOML)."""
        try:
            kwargs = self._collect()
            # ThreadSafeConfig.update(**kwargs)
            self.cfg_holder.update(**kwargs)

            if self.on_apply:
                # Pass the updated EyeTrackerConfig instance
                self.on_apply(self.cfg_holder.get())

            QMessageBox.information(self, "Applied", "Settings applied (not saved).")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply config:\n{e}")

    def save(self):
        """
        Apply then persist to TOML.
        This mirrors the Allied config pattern: we call save_config_section
        with the ThreadSafeConfig.
        """
        try:
            self.apply()
            save_config_section(PUPIL_FIT_TOML_PATH, "eye_tracker", self.cfg_holder)
            QMessageBox.information(
                self,
                "Saved",
                f"Saved to {PUPIL_FIT_TOML_PATH.name}.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")


if __name__ == "__main__":
    import sys

    # Load existing config and wrap in ThreadSafeConfig
    holder = ThreadSafeConfig(
        load_eye_tracker_config(PUPIL_FIT_TOML_PATH, "eye_tracker")
    )
    app = QApplication(sys.argv)
    dlg = EyeTrackerConfigDialog(cfg_holder=holder)
    dlg.show()
    sys.exit(app.exec_())
