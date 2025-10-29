# tools/pupil_fit_config_gui.py
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QApplication, QMessageBox, QCheckBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from brillouin_system.eye_tracker.pupil_fitting.pupil_fitting_config.pupil_fit_config import (
    left_eye_pupil_fit_config, right_eye_pupil_fit_config,
    save_config_section, PUPIL_FIT_TOML_PATH, PupilFitConfig
)

def _float_field() -> QLineEdit:
    e = QLineEdit(); e.setValidator(QDoubleValidator(-1e6, 1e6, 6)); return e
def _int_field() -> QLineEdit:
    e = QLineEdit(); e.setValidator(QIntValidator(0, 99999)); return e

class PupilFitConfigDialog(QDialog):
    def __init__(self, on_apply=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pupil Fit Configuration")
        self.on_apply = on_apply

        self.left_inputs = {}
        self.right_inputs = {}

        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self._group("Left Eye", self.left_inputs))
        row.addWidget(self._group("Right Eye", self.right_inputs))
        layout.addLayout(row)
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self._load()

    def _group(self, label, inputs):
        g = QGroupBox(label)
        v = QVBoxLayout()

        def add_row(field, widget_factory):
            h = QHBoxLayout()
            h.addWidget(QLabel(field.replace("_", " ").capitalize()))
            w = widget_factory()
            inputs[field] = w
            h.addWidget(w)
            v.addLayout(h)

        # fields
        add_row("gaussian_ksize", _int_field)
        add_row("use_otsu", lambda: QCheckBox())
        add_row("close_kernel", _int_field)
        add_row("close_iterations", _int_field)
        add_row("min_area_frac", _float_field)
        add_row("max_area_frac", _float_field)
        add_row("max_bbox_aspect", _float_field)
        add_row("scale", _float_field)

        # ROI
        h = QHBoxLayout()
        h.addWidget(QLabel("ROI (x,y,w,h)"))
        w = QLineEdit(); w.setPlaceholderText("e.g. 120, 90, 220, 160  (leave empty for None)")
        inputs["roi"] = w
        h.addWidget(w)
        v.addLayout(h)

        g.setLayout(v)
        return g

    def _buttons(self):
        h = QHBoxLayout()
        apply_btn = QPushButton("Apply"); apply_btn.clicked.connect(self.apply)
        save_btn = QPushButton("Save");  save_btn.clicked.connect(self.save)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.close)
        h.addStretch(); h.addWidget(apply_btn); h.addWidget(save_btn); h.addWidget(close_btn)
        return h

    def _set_fields(self, inputs, cfg: PupilFitConfig):
        inputs["gaussian_ksize"].setText(str(cfg.gaussian_ksize))
        inputs["use_otsu"].setChecked(bool(cfg.use_otsu))
        inputs["close_kernel"].setText(str(cfg.close_kernel))
        inputs["close_iterations"].setText(str(cfg.close_iterations))
        inputs["min_area_frac"].setText(str(cfg.min_area_frac))
        inputs["max_area_frac"].setText(str(cfg.max_area_frac))
        inputs["max_bbox_aspect"].setText(str(cfg.max_bbox_aspect))
        inputs["scale"].setText(str(cfg.scale))
        if cfg.roi is None:
            inputs["roi"].setText("")
        else:
            inputs["roi"].setText(", ".join(str(x) for x in cfg.roi))

    def _parse_roi(self, text: str):
        s = [t.strip() for t in text.split(",") if t.strip() != ""]
        if not s:
            return None
        if len(s) != 4:
            raise ValueError("ROI must have 4 integers: x,y,w,h or be empty")
        return tuple(int(v) for v in s)

    def _collect(self, inputs) -> dict:
        return {
            "gaussian_ksize": int(inputs["gaussian_ksize"].text() or 5),
            "use_otsu": bool(inputs["use_otsu"].isChecked()),
            "close_kernel": int(inputs["close_kernel"].text() or 3),
            "close_iterations": int(inputs["close_iterations"].text() or 1),
            "min_area_frac": float(inputs["min_area_frac"].text() or 5e-4),
            "max_area_frac": float(inputs["max_area_frac"].text() or 0.5),
            "max_bbox_aspect": float(inputs["max_bbox_aspect"].text() or 3.5),
            "scale": float(inputs["scale"].text() or 0.5),
            "roi": self._parse_roi(inputs["roi"].text()),
        }

    def _load(self):
        left = left_eye_pupil_fit_config.get()
        right = right_eye_pupil_fit_config.get()
        self._set_fields(self.left_inputs, left)
        self._set_fields(self.right_inputs, right)

    def apply(self):
        try:
            left_kwargs = self._collect(self.left_inputs)
            right_kwargs = self._collect(self.right_inputs)
            left_eye_pupil_fit_config.update(**left_kwargs)
            right_eye_pupil_fit_config.update(**right_kwargs)

            if self.on_apply:
                self.on_apply(left_eye_pupil_fit_config.get(), right_eye_pupil_fit_config.get())
            QMessageBox.information(self, "Applied", "Settings applied (not saved).")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply config:\n{e}")

    def save(self):
        try:
            self.apply()
            save_config_section(PUPIL_FIT_TOML_PATH, "left_eye", left_eye_pupil_fit_config)
            save_config_section(PUPIL_FIT_TOML_PATH, "right_eye", right_eye_pupil_fit_config)
            QMessageBox.information(self, "Saved", f"Saved to {PUPIL_FIT_TOML_PATH.name}.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    dlg = PupilFitConfigDialog()
    dlg.show()
    sys.exit(app.exec_())
