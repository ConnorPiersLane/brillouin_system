from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QMessageBox, QSlider
)
from PyQt5.QtGui import QIntValidator

from brillouin_system.config.flir_config.flir_config import (
    _min_gain, _max_gain,
    _min_exposure_time, _max_exposure_time,
    _min_gamma, _max_gamma,
    flir_config, save_flir_settings, flir_config_toml_path
)


class FLIRConfigDialog(QDialog):
    def __init__(self,
                 flir_update_config,
                 parent=None):
        """
        Args:
            flir_update_config = pyqtSignal(object)
        """
        super().__init__(parent)
        self.setWindowTitle("FLIR Camera Settings")
        self.setMinimumSize(360, 320)

        self.flir_update_config = flir_update_config

        self.inputs = {}
        self.slider_controls = {}
        self.current_slider_values = {}

        # ---- ROI + Format controls ----
        form_layout = QVBoxLayout()
        self.inputs["offset_x"] = QLineEdit()
        self.inputs["offset_y"] = QLineEdit()
        self.inputs["width"] = QLineEdit()
        self.inputs["height"] = QLineEdit()
        self.inputs["pixel_format"] = QComboBox()
        self.inputs["pixel_format"].addItems(["Mono8", "Mono16", "RGB8"])  # Example formats

        for key in ["offset_x", "offset_y", "width", "height"]:
            self.inputs[key].setValidator(QIntValidator(0, 9999))

        for key, widget in self.inputs.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(key.replace("_", " ").capitalize() + ":"))
            row.addWidget(widget)
            form_layout.addLayout(row)

        # ---- Slider Controls ----
        self.slider_layout = QVBoxLayout()
        self.add_camera_slider("Exposure Time (µs)", "exposure_time", int(_min_exposure_time), int(_max_exposure_time), 20000)
        self.add_camera_slider("Gain", "gain", int(_min_gain), int(_max_gain), 0)
        self.add_camera_slider("Gamma", "gamma", int(_min_gamma * 100), int(_max_gamma * 100), 100, scale=100)

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

        # ---- Assemble Layout ----
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

        slider.valueChanged.connect(lambda val: self.on_slider_changed(key, label_text, val, label, scale))

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

        # Store initial value
        self.current_slider_values[key] = initial / scale if key == "gamma" else initial

    def on_slider_changed(self, key, label_text, val, label, scale):
        real_val = val / scale
        label.setText(f"{label_text}: {real_val:.2f}")
        self.current_slider_values[key] = real_val if key == "gamma" else val

    def load_values(self):
        cfg = flir_config.get()
        self.inputs["offset_x"].setText(str(cfg.offset_x))
        self.inputs["offset_y"].setText(str(cfg.offset_y))
        self.inputs["width"].setText(str(cfg.width))
        self.inputs["height"].setText(str(cfg.height))
        idx = self.inputs["pixel_format"].findText(cfg.pixel_format)
        if idx >= 0:
            self.inputs["pixel_format"].setCurrentIndex(idx)

    def apply_settings(self):
        try:
            offset_x = int(self.inputs["offset_x"].text())
            offset_y = int(self.inputs["offset_y"].text())
            width = int(self.inputs["width"].text())
            height = int(self.inputs["height"].text())
            pixel_format = self.inputs["pixel_format"].currentText()
            gain = self.slider_controls["gain"].value()
            exposure = self.slider_controls["exposure_time"].value()
            gamma = self.slider_controls["gamma"].value() / 100  # scale down

            # Update config
            flir_config.update(
                offset_x=offset_x,
                offset_y=offset_y,
                width=width,
                height=height,
                pixel_format=pixel_format,
                gain=gain,
                exposure=exposure,
                gamma=gamma,
            )

            self.flir_update_config.emit(flir_config.get())
            print("[FLIR Config] Settings applied.")

        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply settings:\n{e}")

    def save_config(self):
        try:
            gain = self.slider_controls["gain"].value()
            exposure = self.slider_controls["exposure_time"].value()
            gamma = self.slider_controls["gamma"].value() / 100  # scale down

            flir_config.update(
                offset_x=int(self.inputs["offset_x"].text()),
                offset_y=int(self.inputs["offset_y"].text()),
                width=int(self.inputs["width"].text()),
                height=int(self.inputs["height"].text()),
                pixel_format=self.inputs["pixel_format"].currentText(),
                gain=gain,
                exposure=exposure,
                gamma=gamma,
            )

            save_flir_settings(flir_config_toml_path, flir_config)
            print("FLIR config saved.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import pyqtSignal, QObject

    class DummySignals(QObject):
        flir_update_config = pyqtSignal(object)

    app = QApplication(sys.argv)
    signals = DummySignals()

    # Connect test signals to print
    signals.flir_update_config.connect(
        lambda cfg: print(f"[Signal] Config updated: {cfg}")
    )

    dialog = FLIRConfigDialog(
        flir_update_config=signals.flir_update_config
    )
    dialog.exec_()
