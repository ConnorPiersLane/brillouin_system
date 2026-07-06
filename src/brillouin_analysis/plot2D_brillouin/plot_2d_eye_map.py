import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox


def load_excel_file() -> str | None:
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Brillouin Excel Export",
        "",
        "Excel Files (*.xlsx *.xls)"
    )

    if not file_path:
        return None

    return file_path


def validate_columns(df: pd.DataFrame):
    required = ["x_mm", "y_mm", "distance_ghz"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only rows that have valid plotting values
    df = df.dropna(subset=["x_mm", "y_mm", "distance_ghz"])

    # Convert to numeric in case Excel mixed types
    df["x_mm"] = pd.to_numeric(df["x_mm"], errors="coerce")
    df["y_mm"] = pd.to_numeric(df["y_mm"], errors="coerce")
    df["distance_ghz"] = pd.to_numeric(df["distance_ghz"], errors="coerce")

    df = df.dropna(subset=["x_mm", "y_mm", "distance_ghz"])

    if df.empty:
        raise ValueError("No valid rows found with x_mm, y_mm, and distance_ghz.")

    # Average repeated points at the same x/y position
    df_grouped = (
        df.groupby(["x_mm", "y_mm"], as_index=False)["distance_ghz"]
        .mean()
        .sort_values(["x_mm", "y_mm"])
    )

    return df_grouped


def draw_eye_guides(ax):
    # Main pupil / cornea-like guide circles, matching your example look
    radii = [1.0, 2.0, 3.0]
    for r in radii:
        circle = plt.Circle((0, 0), r, fill=False, color="0.25", linewidth=1.1)
        ax.add_patch(circle)

    # Crosshair
    ax.axhline(0, color="0.75", linewidth=0.9)
    ax.axvline(0, color="0.75", linewidth=0.9)

    # Diagonal guides
    diag = 3 / np.sqrt(2)
    ax.plot([-diag, diag], [-diag, diag], color="0.82", linewidth=0.8)
    ax.plot([-diag, diag], [diag, -diag], color="0.82", linewidth=0.8)

    # Radius labels on +x axis
    ax.text(1.02, 0.08, "1 mm", color="0.35", fontsize=9)
    ax.text(2.02, 0.08, "2 mm", color="0.35", fontsize=9)
    ax.text(2.78, 0.08, "3 mm", color="0.35", fontsize=9)


def plot_distance_map(df: pd.DataFrame, title: str = "Distance peak map"):
    x = df["x_mm"].to_numpy()
    y = df["y_mm"].to_numpy()
    z = df["distance_ghz"].to_numpy()

    if len(df) < 3:
        raise ValueError("Need at least 3 valid points to create a filled map.")

    fig, ax = plt.subplots(figsize=(7, 6))

    # Filled triangulated contour
    # --- Settings ---
    vmin = 5.6
    vmax = 5.8
    line_levels = np.arange(vmin, vmax + 0.001, 0.02)
    fill_levels = np.linspace(vmin, vmax, 256)

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_under("navy")
    cmap.set_over("darkred")

    contourf = ax.tricontourf(
        x, y, z,
        levels=fill_levels,
        cmap=cmap,
        extend="both"
    )

    ax.tricontour(
        x, y, z,
        levels=line_levels,
        colors="k",
        linewidths=0.6,
        alpha=0.35
    )


    # Measurement points
    ax.scatter(x, y, s=30, facecolors="white", edgecolors="0.3", zorder=3)

    draw_eye_guides(ax)

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)

    cbar = fig.colorbar(
        contourf,
        ax=ax,
        ticks=line_levels,
        extend="both"
    )
    cbar.set_label("Shift [GHz]")

    plt.tight_layout()
    plt.show()


def main():
    try:
        file_path = load_excel_file()
        if not file_path:
            return

        df = pd.read_excel(file_path)
        validate_columns(df)
        df_plot = prepare_data(df)
        plot_distance_map(df_plot, title="Distance peak map")

    except Exception as e:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        QMessageBox.critical(None, "Plot Failed", str(e))


if __name__ == "__main__":
    main()