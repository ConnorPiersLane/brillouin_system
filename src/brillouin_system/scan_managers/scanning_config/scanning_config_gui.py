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
    """
    Pattern matches AndorConfigDialog:
      - Apply updates in-memory config, then calls a provided callback
      - Save updates in-memory config, writes TOML (optionally also calls callback)
    """

    def __init__(
        self,
        cfg_holder: ThreadSafeConfig | None = None,
        on_apply=None,   # <-- callback: on_apply(ScanningConfig)
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Axial Scanning Configuration")
        self.setMinimumSize(360, 320)

        self.cfg_holder: ThreadSafeConfig = cfg_holder or axial_scanning_config
        self.on_apply = on_apply

        self.inputs: dict[str, QLineEdit] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._group_find_reflection())
        layout.addWidget(self._group_scan_settings())
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self.load_values()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _add_row(self, layout: QVBoxLayout, label: str, key: str, widget: QLineEdit) -> None:
        self.inputs[key] = widget
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(widget, 1)
        layout.addLayout(row)

    def _group_find_reflection(self) -> QGroupBox:
        g = QGroupBox("Find Reflection Settings")
        v = QVBoxLayout()

        le_exposure = QLineEdit()
        le_exposure.setValidator(QDoubleValidator(0.0, 1e9, 6))
        self._add_row(v, "Exposure [s]", "exposure", le_exposure)

        le_gain = QLineEdit()
        le_gain.setValidator(QIntValidator(0, 1_000_000))
        self._add_row(v, "Gain", "gain", le_gain)

        le_n_sigma = QLineEdit()
        le_n_sigma.setValidator(QIntValidator(0, 1_000_000))
        self._add_row(v, "N Sigma", "n_sigma", le_n_sigma)

        le_speed = QLineEdit()
        le_speed.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Speed [µm/s]", "speed_um_s", le_speed)

        le_max_search = QLineEdit()
        le_max_search.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Max search distance [µm]", "max_search_distance_um", le_max_search)

        le_n_bg = QLineEdit()
        le_n_bg.setValidator(QIntValidator(0, 1_000_000))
        self._add_row(v, "N BG images", "n_bg_images", le_n_bg)

        g.setLayout(v)
        return g

    def _group_scan_settings(self) -> QGroupBox:
        g = QGroupBox("Scan Settings")
        v = QVBoxLayout()

        le_max_scan = QLineEdit()
        le_max_scan.setValidator(QIntValidator(0, 10_000_000))
        self._add_row(v, "Max scan distance [µm]", "max_scan_distance_um", le_max_scan)

        g.setLayout(v)
        return g

    def _buttons(self) -> QHBoxLayout:
        h = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)  # <-- andor-like

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)

        h.addStretch()
        h.addWidget(apply_btn)
        h.addWidget(save_btn)
        h.addWidget(close_btn)
        return h

    # ------------------------------------------------------------------ #
    # Data <-> UI (andor-like naming)
    # ------------------------------------------------------------------ #

    def load_values(self) -> None:
        cfg: ScanningConfig = self.cfg_holder.get()
        self.inputs["exposure"].setText(str(cfg.exposure))
        self.inputs["gain"].setText(str(cfg.gain))
        self.inputs["n_sigma"].setText(str(cfg.n_sigma))
        self.inputs["speed_um_s"].setText(str(cfg.speed_um_s))
        self.inputs["max_search_distance_um"].setText(str(cfg.max_search_distance_um))
        self.inputs["n_bg_images"].setText(str(cfg.n_bg_images))
        self.inputs["max_scan_distance_um"].setText(str(cfg.max_scan_distance_um))

    def _update_config_from_inputs(self) -> None:
        """
        Mirrors AndorConfigDialog._update_config_from_inputs():
        - Parse widgets
        - Update ThreadSafeConfig via .update(...)
        """
        def _req_text(key: str) -> str:
            # Keep behavior strict (like Andor: int(...) directly)
            return self.inputs[key].text().strip()

        self.cfg_holder.update(
            exposure=float(_req_text("exposure")),
            gain=int(_req_text("gain")),
            n_sigma=int(_req_text("n_sigma")),
            speed_um_s=float(_req_text("speed_um_s")),
            max_search_distance_um=float(_req_text("max_search_distance_um")),
            n_bg_images=int(_req_text("n_bg_images")),
            max_scan_distance_um=int(_req_text("max_scan_distance_um")),
        )

    # ------------------------------------------------------------------ #
    # Actions (andor-like)
    # ------------------------------------------------------------------ #

    def apply_settings(self) -> None:
        """
        Andor-style Apply:
          - Update in-memory config
          - Call callback with the updated config (so caller can 'send it')
        """
        try:
            self._update_config_from_inputs()

            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())

            print("[Axial Scanning Config] Settings applied.")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply settings:\n{e}")

    def save_config(self) -> None:
        """
        Save:
          - Update in-memory config
          - Persist to TOML
          - (Optional) call callback too (common in hardware UIs)
        """
        try:
            self._update_config_from_inputs()
            save_config_section(AXIAL_SCANNING_TOML_PATH, "axial_scanning", self.cfg_holder)

            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())

            print("[Axial Scanning Config] Settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings:\n{e}")


if __name__ == "__main__":
    import sys

    def example_send_to_system(cfg: ScanningConfig):
        print("[Function Call] Config updated:\n", cfg)

    holder = ThreadSafeConfig(load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH, "axial_scanning"))

    app = QApplication(sys.argv)
    dlg = AxialScanningConfigDialog(cfg_holder=holder, on_apply=example_send_to_system)
    dlg.exec_()
