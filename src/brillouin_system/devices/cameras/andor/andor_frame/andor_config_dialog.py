from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QMessageBox, QApplication
)
from PyQt5.QtGui import QIntValidator

from brillouin_system.devices.cameras.andor.andor_frame.andor_config import (
    andor_frame_config, save_andor_frame_settings, andor_config_toml_path
)


class AndorConfigDialog(QDialog):
    def __init__(self, andor_update_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Andor Camera Settings")
        self.setMinimumSize(300, 400)

        self.andor_update_config = andor_update_config

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
            "n_dark_images": QLineEdit(),
            "n_bg_images": QLineEdit(),
        }

        # Int validators
        for key in ["x_start", "x_end", "y_start", "y_end", "vbin", "hbin", "pre_amp_mode", "vss_index", "n_dark_images", "n_bg_images"]:
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

        # Buttons
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close_dialog)

        btn_layout.addStretch()
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)
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
        self.inputs["n_dark_images"].setText(str(cfg.n_dark_images))
        self.inputs["n_bg_images"].setText(str(cfg.n_bg_images))

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
            n_dark_images=int(self.inputs["n_dark_images"].text()),
            n_bg_images=int(self.inputs["n_bg_images"].text()),
        )

    def apply_settings(self):
        try:
            self._update_config_from_inputs()
            self.andor_update_config(andor_frame_config.get())
            print("[Andor Config] Settings applied.")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply settings:\n{e}")

    def save_config(self):
        try:
            self._update_config_from_inputs()
            save_andor_frame_settings(andor_config_toml_path, andor_frame_config)
            print("[Andor Config] Settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings:\n{e}")

    def close_dialog(self):
        self.close()


if __name__ == "__main__":
    import sys
    def andor_update_config(cfg):
        print("[Function Call] Config updated:\n", cfg)

    app = QApplication(sys.argv)
    dlg = AndorConfigDialog(andor_update_config=andor_update_config)
    dlg.exec_()
