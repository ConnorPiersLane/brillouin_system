from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox,
    QPushButton, QApplication
)


class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Camera Configuration")
        self.setMinimumSize(500, 400)
        layout = QVBoxLayout()

        # --- Spectrum Fitting Group ---
        spectrum_group = QGroupBox("Spectrum Fitting")
        spectrum_layout = QVBoxLayout()

        self.window_size_label = QLabel("Fitting Window Size (rows):")
        self.window_size_input = QLineEdit()

        self.fit_threshold_label = QLabel("Fit Threshold:")
        self.fit_threshold_input = QLineEdit()

        self.sample_label = QGroupBox("Sample:"

        self.measurement_label = QLabel("Measurement:")
        self.measurement_input = QLineEdit()

        spectrum_layout.addWidget(self.window_size_label)
        spectrum_layout.addWidget(self.window_size_input)
        spectrum_layout.addWidget(self.fit_threshold_label)
        spectrum_layout.addWidget(self.fit_threshold_input)
        spectrum_layout.addWidget(self.sample_label)
        spectrum_layout.addWidget(self.sample_input)
        spectrum_layout.addWidget(self.measurement_label)
        spectrum_layout.addWidget(self.measurement_input)

        spectrum_group.setLayout(spectrum_layout)
        layout.addWidget(spectrum_group)

        # --- Buttons ---
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.cancel_button)

        layout.addLayout(button_row)
        self.setLayout(layout)

    def get_settings(self):
        return {
            "spectrum_window_size": self.window_size_input.text(),
            "spectrum_fit_threshold": self.fit_threshold_input.text(),
            "sample": self.sample_input.text(),
            "measurement": self.measurement_input.text()
        }


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    dialog = ConfigDialog()
    if dialog.exec_():
        print("Settings:", dialog.get_settings())
    else:
        print("Dialog cancelled.")
