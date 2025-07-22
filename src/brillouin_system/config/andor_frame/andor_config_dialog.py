from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QMessageBox, QApplication
)
from PyQt5.QtGui import QIntValidator
from brillouin_system.config.andor_frame.andor_config import (
    andor_frame_config, save_andor_frame_settings, AndorConfig
)
from brillouin_system.config.andor_frame.andor_config import andor_config_toml_path


class AndorConfigDialog(QDialog):
    config_updated = pyqtSignal(object)  # Emits AndorConfig (as object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Andor Camera Settings")
        self.setMinimumSize(300, 300)

        self.inputs = {
            "advanced_gain_option": QCheckBox("Use Advanced Gain"),
            "x_start": QLineEdit(),
            "x_end": QLineEdit(),
            "y_start": QLineEdit(),
            "y_end": QLineEdit(),
            "vbin": QLineEdit(),
            "hbin": QLineEdit(),
            "pre_amp_mode": QLineEdit(),
            "vss_index": QLineEdit(),
            "temperature": QLineEdit(),
            "flip_image_horizontally": QCheckBox("Flip Image Horizontally"),
            "verbose": QCheckBox("Verbose Output"),
        }

        for key in ["x_start", "x_end", "y_start", "y_end", "vbin", "hbin", "pre_amp_mode", "vss_index"]:
            self.inputs[key].setValidator(QIntValidator(0, 9999))

        layout = QVBoxLayout()
        for key, widget in self.inputs.items():
            if isinstance(widget, QCheckBox):
                layout.addWidget(widget)
            else:
                row = QHBoxLayout()
                row.addWidget(QLabel(key.replace("_", " ").capitalize() + ":"))
                row.addWidget(widget)
                layout.addLayout(row)

        # Save button
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)

        # Save & emit button
        emit_btn = QPushButton("Save & Emit")
        emit_btn.clicked.connect(self.save_and_emit)
        layout.addWidget(emit_btn)

        self.setLayout(layout)
        self.load_values()

    def load_values(self):
        cfg = andor_frame_config.get()
        self.inputs["advanced_gain_option"].setChecked(cfg.advanced_gain_option)
        self.inputs["x_start"].setText(str(cfg.x_start))
        self.inputs["x_end"].setText(str(cfg.x_end))
        self.inputs["y_start"].setText(str(cfg.y_start))
        self.inputs["y_end"].setText(str(cfg.y_end))
        self.inputs["vbin"].setText(str(cfg.vbin))
        self.inputs["hbin"].setText(str(cfg.hbin))
        self.inputs["pre_amp_mode"].setText(str(cfg.pre_amp_mode))
        self.inputs["vss_index"].setText(str(cfg.vss_index))
        self.inputs["temperature"].setText(str(cfg.temperature))
        self.inputs["flip_image_horizontally"].setChecked(cfg.flip_image_horizontally)
        self.inputs["verbose"].setChecked(cfg.verbose)

    def _parse_temperature(self, value: str) -> float | str:
        value = value.strip().lower()
        if value == "off":
            return value
        try:
            return float(value)
        except ValueError:
            return "off"

    def _update_config_from_inputs(self):
        andor_frame_config.update(
            advanced_gain_option=self.inputs["advanced_gain_option"].isChecked(),
            x_start=int(self.inputs["x_start"].text()),
            x_end=int(self.inputs["x_end"].text()),
            y_start=int(self.inputs["y_start"].text()),
            y_end=int(self.inputs["y_end"].text()),
            vbin=int(self.inputs["vbin"].text()),
            hbin=int(self.inputs["hbin"].text()),
            pre_amp_mode=int(self.inputs["pre_amp_mode"].text()),
            vss_index=int(self.inputs["vss_index"].text()),
            temperature=self._parse_temperature(self.inputs["temperature"].text()),
            flip_image_horizontally=self.inputs["flip_image_horizontally"].isChecked(),
            verbose=self.inputs["verbose"].isChecked(),
        )

    def save_config(self):
        try:
            self._update_config_from_inputs()
            save_andor_frame_settings(andor_config_toml_path, andor_frame_config)
            print("Settings saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def save_and_emit(self):
        try:
            self._update_config_from_inputs()
            save_andor_frame_settings(andor_config_toml_path, andor_frame_config)
            cfg_copy = andor_frame_config.get()
            self.config_updated.emit(cfg_copy)
            print("Settings saved and emitted.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save/emit: {e}")


# Optional test stub
if __name__ == "__main__":
    import sys

    def on_config_received(config):
        print("[Signal] Config received:\n", config)

    app = QApplication(sys.argv)
    dlg = AndorConfigDialog()
    dlg.config_updated.connect(on_config_received)
    dlg.exec_()
