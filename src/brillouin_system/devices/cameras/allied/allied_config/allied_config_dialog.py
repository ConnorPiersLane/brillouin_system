from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QMessageBox, QSlider
)
from PyQt5.QtGui import QIntValidator

from brillouin_system.devices.cameras.allied.allied_config.allied_config import (
    allied_config, save_allied_settings, allied_config_toml_path
)

class AlliedConfigDialog(QDialog):
    def __init__(self, cam_id: str, allied_update_config, parent=None):
        """
        Args:
            cam_id: "left" or "right"
            allied_update_config: pyqtSignal(object)
        """
        super().__init__(parent)
        self.setWindowTitle(f"Allied Vision Camera Settings ({cam_id})")
        self.setMinimumSize(360, 320)

        self.cam_id = cam_id
        self.allied_update_config = allied_update_config

        self.inputs = {}
        self.slider_controls = {}
        self.current_slider_values = {}

        form_layout = QVBoxLayout()
        self.inputs["id"] = QLineEdit()
        self.inputs["offset_x"] = QLineEdit()
        self.inputs["offset_y"] = QLineEdit()
        self.inputs["width"] = QLineEdit()
        self.inputs["height"] = QLineEdit()

        for key in ["offset_x", "offset_y", "width", "height"]:
            self.inputs[key].setValidator(QIntValidator(0, 9999))

        for key, widget in self.inputs.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(key.replace("_", " ").capitalize() + ":"))
            row.addWidget(widget)
            form_layout.addLayout(row)

        # ---- Sliders ----
        self.slider_layout = QVBoxLayout()
        cfg = allied_config[self.cam_id].get()

        self.add_camera_slider("Exposure Time (µs)", "exposure", 10, 1000000, int(cfg.exposure))
        self.add_camera_slider("Gain (dB)", "gain", 0, 50, int(cfg.gain))
        self.add_camera_slider("Gamma", "gamma", 10, 400, int(cfg.gamma * 100), scale=100)

        # ---- Buttons ----
        button_row = QHBoxLayout()
        button_row.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        button_row.addWidget(apply_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)
        button_row.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(close_btn)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addSpacing(10)
        layout.addLayout(self.slider_layout)
        layout.addSpacing(10)
        layout.addLayout(button_row)
        self.setLayout(layout)

        self.load_values()

    def add_camera_slider(self, label_text, key, min_val, max_val, initial, scale=1):
        label = QLabel(f"{label_text}: {initial / scale:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial)
        slider.setTickInterval(max((max_val - min_val) // 10, 1))
        slider.setTickPosition(QSlider.TicksBelow)

        min_lbl = QLabel(str(min_val / scale))
        max_lbl = QLabel(str(max_val / scale))

        slider.valueChanged.connect(
            lambda val: self.on_slider_changed(key, label_text, val, label, scale)
        )

        layout = QVBoxLayout()
        layout.addWidget(label)

        limits = QHBoxLayout()
        limits.addWidget(min_lbl)
        limits.addStretch()
        limits.addWidget(max_lbl)

        layout.addLayout(limits)
        layout.addWidget(slider)

        self.slider_layout.addLayout(layout)
        self.slider_controls[key] = slider
        self.current_slider_values[key] = initial / scale if key == "gamma" else initial

    def on_slider_changed(self, key, label_text, val, label, scale):
        real_val = val / scale
        label.setText(f"{label_text}: {real_val:.2f}")
        self.current_slider_values[key] = real_val if key == "gamma" else val

    def load_values(self):
        cfg = allied_config[self.cam_id].get()
        self.inputs["id"].setText(cfg.id)
        self.inputs["offset_x"].setText(str(cfg.offset_x))
        self.inputs["offset_y"].setText(str(cfg.offset_y))
        self.inputs["width"].setText(str(cfg.width))
        self.inputs["height"].setText(str(cfg.height))

    def apply_settings(self):
        try:
            id_str = self.inputs["id"].text()
            offset_x = int(self.inputs["offset_x"].text())
            offset_y = int(self.inputs["offset_y"].text())
            width = int(self.inputs["width"].text())
            height = int(self.inputs["height"].text())
            gain = self.slider_controls["gain"].value()
            exposure = self.slider_controls["exposure"].value()
            gamma = self.slider_controls["gamma"].value() / 100

            allied_config[self.cam_id].update(
                id=id_str,
                offset_x=offset_x,
                offset_y=offset_y,
                width=width,
                height=height,
                gain=gain,
                exposure=exposure,
                gamma=gamma,
            )

            self.allied_update_config(allied_config[self.cam_id].get())
            print(f"[Allied Config] Settings applied for {self.cam_id}.")

        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply settings:\n{e}")

    def save_config(self):
        try:
            save_allied_settings(allied_config_toml_path, allied_config)
            print(f"Allied config saved to {allied_config_toml_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    def allied_update_config(cfg):
        print(f"[Function Call] Config updated: {cfg}")

    app = QApplication(sys.argv)
    dialog = AlliedConfigDialog("left", allied_update_config)
    dialog.exec_()
