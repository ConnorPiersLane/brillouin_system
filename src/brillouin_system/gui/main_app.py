
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget
from brillouin_viewer import BrillouinViewer  # Your full viewer class

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Modal Imaging Suite")

        # Set central widget
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Optional: use tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Add Brillouin Viewer tab
        self.brillouin_viewer = BrillouinViewer()
        tabs.addTab(self.brillouin_viewer, "Brillouin Viewer")

        # You can now add other widgets/tabs
        dummy_tab = QLabel("Future module: e.g., Fluorescence Imaging")
        dummy_tab.setAlignment(Qt.AlignCenter)
        tabs.addTab(dummy_tab, "Other Modalities")

        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
