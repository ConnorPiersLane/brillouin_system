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
    def __init__(self, cfg_holder: ThreadSafeConfig | None = None, on_apply=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axial Scanning Configuration")
        self.setMinimumSize(380, 460)

        self.cfg_holder: ThreadSafeConfig = cfg_holder or axial_scanning_config
        self.on_apply = on_apply

        self.inputs: dict[str, QLineEdit] = {}
        self.checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._group_axial_scanning())
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self.load_values()

    # ------------------------------------------------------------------ #
    # UI helpers
    # ------------------------------------------------------------------ #

    def _add_row(self, layout: QVBoxLayout, label: str, key: str, widget: QLineEdit):
        self.inputs[key] = widget
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(widget, 1)
        layout.addLayout(row)

    def _add_check(self, layout: QVBoxLayout, label: str, key: str, cb: QCheckBox):
        self.checks[key] = cb
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(cb, 1)
        layout.addLayout(row)

    def _group_axial_scanning(self) -> QGroupBox:
        g = QGroupBox("Axial Scanning")
        v = QVBoxLayout()

        le_n_sigma = QLineEdit()
        le_n_sigma.setValidator(QIntValidator(0, 1_000_000))
        self._add_row(v, "N Sigma", "n_sigma", le_n_sigma)

        le_speed = QLineEdit()
        le_speed.setValidator(QDoubleValidator(-1e12, 1e12, 6))
        self._add_row(v, "Speed [µm/s]", "speed_um_s", le_speed)

        le_max_search = QLineEdit()
        le_max_search.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Max search distance [µm]", "max_search_distance_um", le_max_search)

        le_bg_ms = QLineEdit()
        le_bg_ms.setValidator(QIntValidator(0, 1_000_000))
        self._add_row(v, "Background acquisition time [ms]", "background_acquisition_time_ms", le_bg_ms)

        le_backstep = QLineEdit()
        le_backstep.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Backstep after search [µm]", "backstep_after_search_um", le_backstep)

        cb_refine = QCheckBox("Enabled")
        self._add_check(v, "Refine", "do_refine", cb_refine)

        le_point_ms = QLineEdit()
        le_point_ms.setValidator(QIntValidator(1, 1_000_000))
        self._add_row(v, "Point acquisition time [ms]", "point_acquisition_time_ms", le_point_ms)

        le_step = QLineEdit()
        le_step.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Refine step [µm]", "step_um", le_step)

        le_range = QLineEdit()
        le_range.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Refine range [µm]", "range_um", le_range)

        # NEW: analysis behavior
        le_n_max = QLineEdit()
        le_n_max.setValidator(QIntValidator(1, 1_000_000))
        self._add_row(v, "Top-N max values to average", "n_max_values", le_n_max)

        g.setLayout(v)
        return g

    def _buttons(self) -> QHBoxLayout:
        h = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)

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
    # Data <-> UI
    # ------------------------------------------------------------------ #

    def load_values(self) -> None:
        cfg: ScanningConfig = self.cfg_holder.get()

        for k, w in self.inputs.items():
            w.setText(str(getattr(cfg, k)))

        self.checks["do_refine"].setChecked(bool(cfg.do_refine))

    def _update_config_from_inputs(self) -> None:
        def _req(key: str) -> str:
            return self.inputs[key].text().strip()

        self.cfg_holder.update(
            n_sigma=int(_req("n_sigma")),
            speed_um_s=float(_req("speed_um_s")),
            max_search_distance_um=float(_req("max_search_distance_um")),
            background_acquisition_time_ms=int(_req("background_acquisition_time_ms")),
            backstep_after_search_um=float(_req("backstep_after_search_um")),
            do_refine=bool(self.checks["do_refine"].isChecked()),
            point_acquisition_time_ms=int(_req("point_acquisition_time_ms")),
            step_um=float(_req("step_um")),
            range_um=float(_req("range_um")),
            n_max_values=max(1, int(_req("n_max_values"))),
        )

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def apply_settings(self) -> None:
        try:
            self._update_config_from_inputs()
            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())
            print("[Axial Scanning Config] Settings applied.")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", str(e))

    def save_config(self) -> None:
        try:
            self._update_config_from_inputs()
            save_config_section(AXIAL_SCANNING_TOML_PATH, "axial_scanning", self.cfg_holder)
            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())
            print("[Axial Scanning Config] Settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))


if __name__ == "__main__":
    import sys

    def example_send(cfg: ScanningConfig):
        print(cfg)

    holder = ThreadSafeConfig(load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH))

    app = QApplication(sys.argv)
    dlg = AxialScanningConfigDialog(holder, example_send)
    dlg.exec_()