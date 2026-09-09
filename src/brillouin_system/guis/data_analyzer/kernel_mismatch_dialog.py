"""The analyzer's answer to a stored-PSF mismatch (kernel_source = "file").

fit_axial_scan compares the stored ePSF table with every scan's own
calibration profile once per scan. Without a GUI a failed line silently
falls back to the scan's own kernel (with a WARNING in the log). In the
data analyzer the user is asked instead: recalculate the kernel of the
failed lines from this scan's calibration (Yes, the default), or keep the
stored PSF (No). The decision is per scan, because the check is.
"""
from PyQt5.QtWidgets import QMessageBox

from brillouin_system.analysis.fit_axial_scan import set_kernel_mismatch_handler


def ask_recalculate(bad, profiles, parent=None) -> bool:
    """Modal question, True = fall back to the scan's own calibration."""
    names = ", ".join(m.name for m in bad)
    lines = "\n".join(str(m) for m in bad)
    answer = QMessageBox.question(
        parent, "Stored PSF does not match this scan",
        f"The loaded PSF table\n{profiles.path}\n"
        f"does not match the calibration measured with this scan on: {names}\n\n"
        f"{lines}\n\n"
        "Recalculate the kernel of these lines from this scan's own "
        "calibration?\nYes = use the scan's calibration (safe), "
        "No = keep the stored PSF.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    return answer == QMessageBox.Yes


def install_kernel_mismatch_dialog(parent=None):
    """Route the mismatch decision to the dialog for this process."""
    set_kernel_mismatch_handler(lambda bad, profiles: ask_recalculate(bad, profiles, parent))
