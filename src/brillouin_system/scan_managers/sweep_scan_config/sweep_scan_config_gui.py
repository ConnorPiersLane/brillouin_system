from __future__ import annotations

from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import (
    SWEEP_SCAN_TOML_PATH,
    SweepScanConfig,
    sweep_scan_config,
    load_sweep_scan_config,
    save_sweep_scan_config,
)


class SweepScanConfigDialog(QDialog):
    def __init__(self, cfg_holder: ThreadSafeConfig | None = None, on_apply=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sweep Scan Configuration")
        self.setMinimumSize(430, 280)

        self.cfg_holder: ThreadSafeConfig = cfg_holder or sweep_scan_config
        self.on_apply = on_apply

        self.inputs: dict[str, QLineEdit] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._group_sweep_scan())
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

    def _group_sweep_scan(self) -> QGroupBox:
        g = QGroupBox("In-Out Sweep Scan")
        v = QVBoxLayout()

        le_n_repeats = QLineEdit()
        le_n_repeats.setValidator(QIntValidator(1, 1_000_000))
        self._add_row(v, "Number of cycles", "n_repeats", le_n_repeats)

        le_approach = QLineEdit()
        le_approach.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Approach past plane [µm]", "approach_um", le_approach)

        le_target_depth = QLineEdit()
        le_target_depth.setValidator(QDoubleValidator(-1e12, 1e12, 6))
        self._add_row(v, "Target depth [µm]", "target_depth_um", le_target_depth)

        le_settle = QLineEdit()
        le_settle.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Settle before snap [s]", "settle_s", le_settle)

        le_gate = QLineEdit()
        le_gate.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "In-crossing gate [µm]", "plausibility_gate_um", le_gate)

        le_out_gate = QLineEdit()
        le_out_gate.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Out-crossing gate [µm]", "out_gate_um", le_out_gate)

        le_min_peak = QLineEdit()
        le_min_peak.setValidator(QDoubleValidator(0.0, 1.0, 6))
        self._add_row(v, "Min peak fraction", "min_peak_fraction", le_min_peak)

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
        cfg: SweepScanConfig = self.cfg_holder.get()

        for k, w in self.inputs.items():
            w.setText(str(getattr(cfg, k)))

    def _update_config_from_inputs(self) -> None:
        def _req(key: str) -> str:
            return self.inputs[key].text().strip()

        self.cfg_holder.update(
            n_repeats=max(1, int(_req("n_repeats"))),
            approach_um=float(_req("approach_um")),
            target_depth_um=float(_req("target_depth_um")),
            settle_s=float(_req("settle_s")),
            plausibility_gate_um=float(_req("plausibility_gate_um")),
            out_gate_um=float(_req("out_gate_um")),
            min_peak_fraction=float(_req("min_peak_fraction")),
        )

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def apply_settings(self) -> None:
        try:
            self._update_config_from_inputs()
            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())
            print("[Sweep Scan Config] Settings applied.")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", str(e))

    def save_config(self) -> None:
        try:
            self._update_config_from_inputs()
            save_sweep_scan_config(self.cfg_holder, SWEEP_SCAN_TOML_PATH, "sweep_scan")
            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())
            print("[Sweep Scan Config] Settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))


if __name__ == "__main__":
    import sys

    def example_send(cfg: SweepScanConfig):
        print(cfg)

    holder = ThreadSafeConfig(load_sweep_scan_config(SWEEP_SCAN_TOML_PATH))

    app = QApplication(sys.argv)
    dlg = SweepScanConfigDialog(holder, example_send)
    dlg.exec_()
