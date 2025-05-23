from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox,
    QPushButton, QApplication
)
from brillouin_system.config.fitting_config import (
    sample_config,
    reference_config,
    sline_config,
    save_find_peaks_config_section,
    save_selected_rows,
    find_peaks_config_toml_path,
)


class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Peak Detection Settings")
        self.setMinimumSize(500, 400)

        self.sample_inputs = {}
        self.reference_inputs = {}

        layout = QVBoxLayout()
        layout.addWidget(self.create_row_input_block())
        layout.addWidget(self.create_spectrum_fitting_group())
        layout.addLayout(self.create_save_button())
        self.setLayout(layout)

        self.load_initial_values()

    def create_row_input_block(self):
        row = QHBoxLayout()
        self.selected_rows_label = QLabel("Selected Pixel Rows (e.g. 4,5,6):")
        self.selected_rows_input = QLineEdit()
        self.selected_rows_input.textChanged.connect(self.on_rows_input_changed)
        row.addWidget(self.selected_rows_label)
        row.addWidget(self.selected_rows_input)
        container = QGroupBox("Rows of Pixel to be summed")
        container.setLayout(row)
        return container

    def create_spectrum_fitting_group(self):
        group = QGroupBox("Spectrum Fitting")
        layout = QVBoxLayout()
        layout.addWidget(self.create_config_group("Sample", sample_config, self.sample_inputs))
        layout.addWidget(self.create_config_group("Reference", reference_config, self.reference_inputs))
        group.setLayout(layout)
        return group

    def create_config_group(self, title, config_obj, inputs_dict):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        for field in self.field_names():
            row = QHBoxLayout()
            label = QLabel(field.replace("_", " ").capitalize() + ":")
            input_field = QLineEdit()
            input_field.textChanged.connect(self.make_handler(config_obj, field))
            inputs_dict[field] = input_field
            row.addWidget(label)
            row.addWidget(input_field)
            layout.addLayout(row)
        group.setLayout(layout)
        return group

    def make_handler(self, config_obj, field):
        def handler(val):
            parsed = self._parse_value(val, field)
            config_obj.set(field, parsed)
        return handler

    def create_save_button(self):
        self.save_button = QPushButton("Save Values")
        self.save_button.clicked.connect(self.on_save_values_clicked)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        bottom_row.addWidget(self.save_button)

        return bottom_row

    def field_names(self):
        return [
            "prominence_fraction",
            "min_peak_width",
            "min_peak_height",
            "rel_height",
            "wlen_pixels",
        ]

    def load_initial_values(self):
        # Load rows
        self.selected_rows_input.setText(",".join(str(x) for x in sline_config.get().selected_rows))

        # Load sample and reference config values
        for field in self.field_names():
            self.sample_inputs[field].setText(str(getattr(sample_config.get(), field)))
            self.reference_inputs[field].setText(str(getattr(reference_config.get(), field)))

    def on_rows_input_changed(self, text: str):
        try:
            rows = [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]
            sline_config.set("selected_rows", rows)
        except ValueError:
            pass  # Optional: display validation feedback

    def on_save_values_clicked(self):
        try:
            save_selected_rows(find_peaks_config_toml_path, sline_config)
            save_find_peaks_config_section(find_peaks_config_toml_path, "sample", sample_config)
            save_find_peaks_config_section(find_peaks_config_toml_path, "reference", reference_config)
            print("[ConfigDialog] Saved successfully.")
        except Exception as e:
            print(f"[ConfigDialog] Error while saving: {e}")

    def _parse_value(self, value: str, field: str):
        value = value.strip()
        if value.lower() == "none":
            return None
        try:
            if "fraction" in field or "rel" in field:
                return float(value)
            return int(value)
        except ValueError:
            return None


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    dialog = ConfigDialog()
    dialog.show()
    sys.exit(app.exec_())
