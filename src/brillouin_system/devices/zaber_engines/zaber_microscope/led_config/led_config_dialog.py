from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt

from brillouin_system.devices.zaber_engines.zaber_microscope.led_config.led_config import (
    led_config, save_led_settings, led_config_toml_path, LEDConfig
)


class LEDConfigDialog(QDialog):
    def __init__(self, update_led_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LED Configuration")
        self.setMinimumSize(400, 300)

        self.update_led_config = update_led_config
        self.sliders = {}
        self.switches = {}
        self.labels = {}

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.led_names = [
            ("white_below", "White Below"),
            ("blue_385_below", "Blue 385 Below"),
            ("red_625_below", "Red 625 Below"),
            ("white_top", "White Top"),
        ]

        self._init_ui()
        self.load_values()

    def _init_ui(self):
        for key, label_text in self.led_names:
            row = QHBoxLayout()

            label = QLabel(f"{label_text}:")
            label.setFixedWidth(110)

            checkbox = QCheckBox()
            self.switches[key] = checkbox

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setFixedWidth(200)
            self.sliders[key] = slider

            value_label = QLabel("100%")
            value_label.setFixedWidth(50)
            self.labels[key] = value_label

            slider.valueChanged.connect(lambda val, k=key: self.labels[k].setText(f"{val}%"))

            row.addWidget(label)
            row.addWidget(checkbox)
            row.addWidget(slider)
            row.addWidget(value_label)
            self.layout.addLayout(row)

        # Buttons
        button_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        button_row.addWidget(apply_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)
        button_row.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(close_btn)

        self.layout.addLayout(button_row)

    def load_values(self):
        cfg = led_config.get()
        self.sliders["white_below"].setValue(cfg.intensity_led_white_below)
        self.sliders["blue_385_below"].setValue(cfg.intensity_led_blue_385_below)
        self.sliders["red_625_below"].setValue(cfg.intensity_led_red_625_below)
        self.sliders["white_top"].setValue(cfg.intensity_led_white_top)

        self.switches["white_below"].setChecked(cfg.is_led_white_below)
        self.switches["blue_385_below"].setChecked(cfg.is_led_blue_385_below)
        self.switches["red_625_below"].setChecked(cfg.is_led_red_625_below)
        self.switches["white_top"].setChecked(cfg.is_led_white_top)

    def apply_settings(self):
        try:
            led_config.update(
                intensity_led_white_below=self.sliders["white_below"].value(),
                intensity_led_blue_385_below=self.sliders["blue_385_below"].value(),
                intensity_led_red_625_below=self.sliders["red_625_below"].value(),
                intensity_led_white_top=self.sliders["white_top"].value(),
                is_led_white_below=self.switches["white_below"].isChecked(),
                is_led_blue_385_below=self.switches["blue_385_below"].isChecked(),
                is_led_red_625_below=self.switches["red_625_below"].isChecked(),
                is_led_white_top=self.switches["white_top"].isChecked(),
            )

            self.update_led_config(led_config.get())
            print("[LED Config] Settings applied.")

        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply LED settings:\n{e}")

    def save_config(self):
        try:
            # Ensure config is updated with current UI state
            led_config.update(
                intensity_led_white_below=self.sliders["white_below"].value(),
                intensity_led_blue_385_below=self.sliders["blue_385_below"].value(),
                intensity_led_red_625_below=self.sliders["red_625_below"].value(),
                intensity_led_white_top=self.sliders["white_top"].value(),
                is_led_white_below=self.switches["white_below"].isChecked(),
                is_led_blue_385_below=self.switches["blue_385_below"].isChecked(),
                is_led_red_625_below=self.switches["red_625_below"].isChecked(),
                is_led_white_top=self.switches["white_top"].isChecked(),
            )

            save_led_settings(led_config_toml_path, led_config)
            print("[LED Config] Configuration saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save LED config:\n{e}")


# ----------------------------- #
# Standalone Test Entry Point
# ----------------------------- #
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    def dummy_update_led_config(cfg: LEDConfig):
        print("[✓] LED Config Applied from Main:")
        print(cfg)

    app = QApplication(sys.argv)
    dialog = LEDConfigDialog(update_led_config=dummy_update_led_config)
    dialog.exec_()
