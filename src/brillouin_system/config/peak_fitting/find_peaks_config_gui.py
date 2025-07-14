# find_peaks_config_dialog.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QApplication
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from brillouin_system.config.peak_fitting.find_peaks_config import (
    find_peaks_sample_config, find_peaks_reference_config,
    save_config_section, FIND_PEAKS_TOML_PATH,
    FITTING_MODELS_SAMPLE, FITTING_MODELS_REFERENCE
)


class FindPeaksConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Peaks Configuration")
        self.setMinimumSize(500, 200)

        self.sample_inputs = {}
        self.reference_inputs = {}

        layout = QVBoxLayout()
        layout.addLayout(self.create_dual_form())
        layout.addWidget(self.create_save_button())
        self.setLayout(layout)

        self.load_values()

    def field_names(self):
        return ["prominence_fraction", "min_peak_width", "min_peak_height", "rel_height", "wlen_pixels"]

    def create_dual_form(self):
        layout = QHBoxLayout()
        layout.addWidget(self.create_config_group("Sample", self.sample_inputs, FITTING_MODELS_SAMPLE))
        layout.addWidget(self.create_config_group("Reference", self.reference_inputs, FITTING_MODELS_REFERENCE))
        return layout

    def create_config_group(self, label, inputs, models):
        group = QGroupBox(label)
        vlayout = QVBoxLayout()
        for field in self.field_names():
            row = QHBoxLayout()
            row.addWidget(QLabel(field.replace("_", " ").capitalize()))
            edit = QLineEdit()
            if "fraction" in field or "rel" in field:
                edit.setValidator(QDoubleValidator(0.0, 1.0, 5))
            else:
                edit.setValidator(QIntValidator(0, 9999))
            inputs[field] = edit
            row.addWidget(edit)
            vlayout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Fitting Model"))
        combo = QComboBox()
        combo.addItems(models)
        inputs["fitting_model"] = combo
        row.addWidget(combo)
        vlayout.addLayout(row)

        group.setLayout(vlayout)
        return group

    def load_values(self):
        sample = find_peaks_sample_config.get()
        reference = find_peaks_reference_config.get()

        for field in self.field_names():
            self.sample_inputs[field].setText(str(getattr(sample, field)))
            self.reference_inputs[field].setText(str(getattr(reference, field)))

        self.sample_inputs["fitting_model"].setCurrentText(sample.fitting_model)
        self.reference_inputs["fitting_model"].setCurrentText(reference.fitting_model)

    def save_config(self):
        sample_kwargs = {f: self._parse(self.sample_inputs[f].text(), f) for f in self.field_names()}
        sample_kwargs["fitting_model"] = self.sample_inputs["fitting_model"].currentText()

        reference_kwargs = {f: self._parse(self.reference_inputs[f].text(), f) for f in self.field_names()}
        reference_kwargs["fitting_model"] = self.reference_inputs["fitting_model"].currentText()

        find_peaks_sample_config.update(**sample_kwargs)
        find_peaks_reference_config.update(**reference_kwargs)

        save_config_section(FIND_PEAKS_TOML_PATH, "sample", find_peaks_sample_config)
        save_config_section(FIND_PEAKS_TOML_PATH, "reference", find_peaks_reference_config)

        print("[FindPeaksConfigDialog] Saved.")

    def create_save_button(self):
        button = QPushButton("Save")
        button.clicked.connect(self.save_config)
        return button

    def _parse(self, value, field):
        value = value.strip()
        try:
            return float(value) if "fraction" in field or "rel" in field else int(value)
        except ValueError:
            return 0

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    dialog = FindPeaksConfigDialog()
    dialog.show()
    sys.exit(app.exec_())
