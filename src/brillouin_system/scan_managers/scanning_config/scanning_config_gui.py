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
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig
from brillouin_system.scan_managers.scanning_config.scanning_config import (
    ScanningConfig,
    AXIAL_SCANNING_TOML_PATH,
    load_axial_scanning_config,
    save_config_section,
    axial_scanning_config,
)


class AxialScanningConfigDialog(QDialog):
    def __init__(
        self,
        cfg_holder: ThreadSafeConfig | None = None,
        on_apply=None,
        parent=None,
    ):
        """
        cfg_holder:
            ThreadSafeConfig[AxialScanningConfig]. If None, the global
            `axial_scanning_config` is used.

        on_apply:
            Optional callback taking a single AxialScanningConfig, called
            after cfg_holder has been updated (but before saving).
        """
        super().__init__(parent)
        self.setWindowTitle("Axial Scanning Configuration")

        self.cfg_holder: ThreadSafeConfig = cfg_holder or axial_scanning_config
        self.on_apply = on_apply

        self.inputs: dict[str, object] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._group_find_reflection(self.inputs))
        layout.addWidget(self._group_scan_settings(self.inputs))
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self._load()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _group_find_reflection(self, inputs: dict) -> QGroupBox:
        g = QGroupBox("Find Reflection Settings")
        v = QVBoxLayout()

        def add_row(label: str, widget: QLineEdit):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget, 1)
            v.addLayout(h)

        # Exposure time
        le_exposure = QLineEdit()
        le_exposure.setValidator(QDoubleValidator(0.0, 1e6, 6))
        inputs["exposure_time"] = le_exposure
        add_row("Exposure time [s]", le_exposure)

        le_gain = QLineEdit()
        le_gain.setValidator(QDoubleValidator(0.0, 1e6, 6))
        inputs["gain"] = le_gain
        add_row("Gain", le_gain)

        # Threshold value
        le_thresh = QLineEdit()
        le_thresh.setValidator(QDoubleValidator(0.0, 1e18, 6))
        inputs["reflection_threshold_value"] = le_thresh
        add_row("Reflection threshold", le_thresh)

        # Step distance [um]
        le_step = QLineEdit()
        le_step.setValidator(QIntValidator(0, 10_000_000))
        inputs["step_distance_um"] = le_step
        add_row("Step distance [um]", le_step)

        # Max distance [um]
        le_max_dist = QLineEdit()
        le_max_dist.setValidator(QIntValidator(0, 10_000_000))
        inputs["max_distance_um"] = le_max_dist
        add_row("Max distance [um]", le_max_dist)

        # Step after finding Reflection [um]
        le_step_after = QLineEdit()
        le_step_after.setValidator(QIntValidator(0, 10_000_000))
        inputs["step_after_finding_reflection_um"] = le_step_after
        add_row("Step after finding Reflection [um]", le_step_after)

        # N BG Images
        le_n_bg = QLineEdit()
        le_n_bg.setValidator(QIntValidator(0, 1_000_000))
        inputs["n_bg_images"] = le_n_bg
        add_row("N BG Images", le_n_bg)

        g.setLayout(v)
        return g

    def _group_scan_settings(self, inputs: dict) -> QGroupBox:
        g = QGroupBox("Scan Settings")
        v = QVBoxLayout()

        def add_row(label: str, widget: QLineEdit):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget, 1)
            v.addLayout(h)

        # Max Scan Dista [um].
        le_max_scan = QLineEdit()
        le_max_scan.setValidator(QIntValidator(0, 10_000_000))
        inputs["max_scan_distance_um"] = le_max_scan
        add_row("Max Scan Dista [um].", le_max_scan)

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

    def _set_fields(self, cfg: ScanningConfig) -> None:
        self.inputs["exposure_time"].setText(str(cfg.exposure_time_for_reflection_finding))
        self.inputs["gain"].setText(str(cfg.gain_for_reflection_finding))
        self.inputs["reflection_threshold_value"].setText(str(cfg.reflection_threshold_value))
        self.inputs["step_distance_um"].setText(str(cfg.step_distance_um_for_reflection_finding))
        self.inputs["max_distance_um"].setText(str(cfg.max_search_distance_um_for_reflection_finding))
        self.inputs["step_after_finding_reflection_um"].setText(
            str(cfg.step_after_finding_reflection_um)
        )
        self.inputs["n_bg_images"].setText(str(cfg.n_bg_images_for_reflection_finding))
        self.inputs["max_scan_distance_um"].setText(str(cfg.max_scan_distance_um))

    def _collect(self) -> dict[str, int]:
        def _intval(key: str, default: int) -> int:
            text = self.inputs[key].text()
            return int(text) if text else default

        def _floatval(key: str, default: float) -> float:
            text = self.inputs[key].text()
            return float(text) if text else default

        return {
            "exposure_time_for_reflection_finding": _floatval("exposure_time", 0.05),
            "gain_for_reflection_finding": _intval("gain", 1),  # <-- ADD
            "reflection_threshold_value": _floatval("reflection_threshold_value", 5000.0),
            "step_distance_um_for_reflection_finding": _intval("step_distance_um", 20),
            "max_search_distance_um_for_reflection_finding": _intval("max_distance_um", 2000),
            "step_after_finding_reflection_um": _intval("step_after_finding_reflection_um", 20),
            "n_bg_images_for_reflection_finding": _intval("n_bg_images", 10),
            "max_scan_distance_um": _intval("max_scan_distance_um", 2000),
        }

    def _load(self) -> None:
        cfg = self.cfg_holder.get()
        self._set_fields(cfg)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def apply(self) -> None:
        """Apply settings to in-memory config (but do not save TOML)."""
        try:
            kwargs = self._collect()
            self.cfg_holder.update(**kwargs)

            if self.on_apply:
                self.on_apply(self.cfg_holder.get())

            QMessageBox.information(self, "Applied", "Settings applied (not saved).")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply config:\n{e}")

    def save(self) -> None:
        """Apply then persist to TOML."""
        try:
            self.apply()
            save_config_section(AXIAL_SCANNING_TOML_PATH, "axial_scanning", self.cfg_holder)
            QMessageBox.information(
                self,
                "Saved",
                f"Saved to {AXIAL_SCANNING_TOML_PATH.name}.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")


if __name__ == "__main__":
    import sys

    holder = ThreadSafeConfig(
        load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH, "axial_scanning")
    )
    app = QApplication(sys.argv)
    dlg = AxialScanningConfigDialog(cfg_holder=holder)
    dlg.show()
    sys.exit(app.exec_())
