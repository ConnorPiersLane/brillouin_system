# stereo_calib_gui.py
from __future__ import annotations

import os
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Optional, List, Tuple

from brillouin_system.eye_tracker.stereo_imaging.calibrate_single import (
    MonoCalibConfig,
    CameraResult,
    calibrate_single,
    save_camera_json,
    detect_corners,
)
from brillouin_system.eye_tracker.stereo_imaging.calibrate_stereo import (
    StereoCalibConfig,
    LeftFramesValid,
    RightFramesValid,
    PairsValid,
    load_image_pairs_smart,
    filter_valid_frames,
    stereo_calibrate_from_pairs,
    save_stereo_json,
)


@dataclass
class AppState:
    folder: str = ""
    cols: int = 10
    rows: int = 8
    square_mm: float = 2.0
    model: str = "pinhole"
    reference: str = "left"
    prefix: str = "calibration"

    # Data containers
    left_valid: Optional[LeftFramesValid] = None
    right_valid: Optional[RightFramesValid] = None
    pair_valid: Optional[PairsValid] = None
    all_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    # Mono calibration results
    left_res: Optional[CameraResult] = None
    right_res: Optional[CameraResult] = None

    # Viewer state
    pair_index: int = 0
    det_cache: Optional[List[Tuple[bool, np.ndarray | None, bool, np.ndarray | None]]] = None


class StereoCalibGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stereo Calibration GUI")
        self.state = AppState()
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Folder
        row = 0
        ttk.Label(root, text="Images Folder:").grid(row=row, column=0, sticky="w")
        self.var_folder = tk.StringVar()
        ttk.Entry(root, textvariable=self.var_folder, width=50).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(root, text="Browse", command=self._on_browse).grid(row=row, column=2)
        root.columnconfigure(1, weight=1)

        # Parameters
        row += 1
        frame = ttk.Frame(root)
        frame.grid(row=row, column=0, columnspan=3, pady=10, sticky="we")
        self.var_cols = tk.IntVar(value=9)
        self.var_rows = tk.IntVar(value=6)
        self.var_size = tk.DoubleVar(value=25.0)
        self.var_model = tk.StringVar(value="pinhole")
        self.var_ref = tk.StringVar(value="left")
        self.var_prefix = tk.StringVar(value="calibration")

        ttk.Label(frame, text="Cols").grid(row=0, column=0)
        ttk.Entry(frame, textvariable=self.var_cols, width=5).grid(row=0, column=1)
        ttk.Label(frame, text="Rows").grid(row=0, column=2)
        ttk.Entry(frame, textvariable=self.var_rows, width=5).grid(row=0, column=3)
        ttk.Label(frame, text="Square (mm)").grid(row=0, column=4)
        ttk.Entry(frame, textvariable=self.var_size, width=8).grid(row=0, column=5)
        ttk.Label(frame, text="Model").grid(row=0, column=6)
        ttk.Combobox(frame, textvariable=self.var_model, values=["pinhole", "fisheye"], width=10, state="readonly").grid(row=0, column=7)
        ttk.Label(frame, text="Reference").grid(row=0, column=8)
        ttk.Combobox(frame, textvariable=self.var_ref, values=["left", "right"], width=8, state="readonly").grid(row=0, column=9)
        ttk.Label(frame, text="Prefix").grid(row=0, column=10)
        ttk.Entry(frame, textvariable=self.var_prefix, width=12).grid(row=0, column=11)

        # Buttons
        row += 1
        bframe = ttk.Frame(root)
        bframe.grid(row=row, column=0, columnspan=3, pady=6)
        ttk.Button(bframe, text="Scan & Validate Frames", command=lambda: self._async(self._scan_and_validate_impl)).grid(row=0, column=0, padx=5)
        ttk.Button(bframe, text="Run Left Mono", command=lambda: self._async(self._run_left_mono_impl)).grid(row=0, column=1, padx=5)
        ttk.Button(bframe, text="Run Right Mono", command=lambda: self._async(self._run_right_mono_impl)).grid(row=0, column=2, padx=5)
        ttk.Button(bframe, text="Run Stereo (Extrinsics Only)", command=lambda: self._async(self._run_stereo_impl)).grid(row=0, column=3, padx=5)

        # Viewer
        row += 1
        self.viewer_frame = ttk.Frame(root)
        self.viewer_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=10)
        root.rowconfigure(row, weight=1)

        ctrl = ttk.Frame(self.viewer_frame)
        ctrl.grid(row=0, column=0, sticky="we", pady=(0, 6))
        self.btn_prev = ttk.Button(ctrl, text="⟵ Prev", command=self._on_prev_pair, state="disabled")
        self.btn_next = ttk.Button(ctrl, text="Next ⟶", command=self._on_next_pair, state="disabled")
        self.lbl_status = ttk.Label(ctrl, text="No pairs loaded")
        self.btn_prev.grid(row=0, column=0, padx=5)
        self.btn_next.grid(row=0, column=1, padx=5)
        self.lbl_status.grid(row=0, column=2, padx=10)
        ctrl.columnconfigure(2, weight=1)

        canv = ttk.Frame(self.viewer_frame)
        canv.grid(row=1, column=0, sticky="nsew")
        self.cnv_left = tk.Canvas(canv, width=480, height=360, bg="#111")
        self.cnv_right = tk.Canvas(canv, width=480, height=360, bg="#111")
        self.cnv_left.grid(row=0, column=0, padx=4, pady=4)
        self.cnv_right.grid(row=0, column=1, padx=4, pady=4)
        canv.columnconfigure(0, weight=1)
        canv.columnconfigure(1, weight=1)

        # Log
        row += 1
        ttk.Label(root, text="Log:").grid(row=row, column=0, sticky="w")
        row += 1
        self.txt_log = tk.Text(root, height=8)
        self.txt_log.grid(row=row, column=0, columnspan=3, sticky="nsew")
        root.rowconfigure(row, weight=1)

    # ---------------- Core ----------------
    def _on_browse(self):
        folder = filedialog.askdirectory()
        if folder:
            self.var_folder.set(folder)

    def _async(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    # ---------------- Operations ----------------
    def _scan_and_validate_impl(self):
        folder = self.var_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            self._log("Invalid folder")
            return
        self._log("Scanning for image pairs…")
        pairs = load_image_pairs_smart(folder)
        if not pairs:
            self._log("No pairs found")
            return
        self.state.all_pairs = pairs
        cols, rows = int(self.var_cols.get()), int(self.var_rows.get())
        L_valid, R_valid, P_valid, report = filter_valid_frames(pairs, cols, rows)
        self.state.left_valid, self.state.right_valid, self.state.pair_valid = L_valid, R_valid, P_valid
        self._log(f"Valid left: {len(L_valid.images)} | right: {len(R_valid.images)} | pairs: {len(P_valid.pairs)}")
        self.state.det_cache = None
        self.state.pair_index = 0
        self._render_pair()

    def _run_left_mono_impl(self):
        if not self.state.left_valid or not self.state.left_valid.images:
            self._log("No left frames detected")
            return
        cfg = MonoCalibConfig(model=self.var_model.get(), cols=int(self.var_cols.get()),
                              rows=int(self.var_rows.get()), square_size_mm=float(self.var_size.get()))
        self._log("Running left mono calibration…")
        res, imgsize = calibrate_single(self.state.left_valid.images, cfg)
        self.state.left_res = res
        out = os.path.join(self.var_folder.get(), f"{self.var_prefix.get()}_left.json")
        save_camera_json(out, res, imgsize, cfg)
        self._log(f"Saved {out}")

    def _run_right_mono_impl(self):
        if not self.state.right_valid or not self.state.right_valid.images:
            self._log("No right frames detected")
            return
        cfg = MonoCalibConfig(model=self.var_model.get(), cols=int(self.var_cols.get()),
                              rows=int(self.var_rows.get()), square_size_mm=float(self.var_size.get()))
        self._log("Running right mono calibration…")
        res, imgsize = calibrate_single(self.state.right_valid.images, cfg)
        self.state.right_res = res
        out = os.path.join(self.var_folder.get(), f"{self.var_prefix.get()}_right.json")
        save_camera_json(out, res, imgsize, cfg)
        self._log(f"Saved {out}")

    def _run_stereo_impl(self):
        if not self.state.pair_valid or not self.state.pair_valid.pairs:
            self._log("No valid pairs found")
            return
        if not self.state.left_res or not self.state.right_res:
            self._log("Run both mono calibrations first!")
            return
        cfg = StereoCalibConfig(model=self.var_model.get(), reference=self.var_ref.get(),
                                cols=int(self.var_cols.get()), rows=int(self.var_rows.get()),
                                square_size_mm=float(self.var_size.get()))
        self._log("Running stereo calibration (extrinsics only)…")
        stereo = stereo_calibrate_from_pairs(self.state.pair_valid, cfg,
                                             self.state.left_res, self.state.right_res)
        h, w = self.state.pair_valid.pairs[0][0].shape[:2]
        out = os.path.join(self.var_folder.get(), f"{self.var_prefix.get()}_stereo.json")
        save_stereo_json(out, stereo, cfg, (w, h))
        self._log(f"Saved {out}")
        messagebox.showinfo("Done", "Stereo calibration complete!")

    # ---------------- Viewer ----------------
    def _ensure_det_cache(self):
        # Use all pairs (including failures) for display
        pairs = self.state.all_pairs
        if not pairs:
            self.state.det_cache = None
            return
        if self.state.det_cache is not None and len(self.state.det_cache) == len(pairs):
            return

        cols, rows = int(self.var_cols.get()), int(self.var_rows.get())
        pattern = (cols, rows)
        cache = []
        for L, R in pairs:
            cL = detect_corners(L, pattern)
            cR = detect_corners(R, pattern)
            cache.append((cL is not None, cL, cR is not None, cR))
        self.state.det_cache = cache

    def _on_prev_pair(self):
        if not self.state.all_pairs:
            return
        self.state.pair_index = max(0, self.state.pair_index - 1)
        self._render_pair()

    def _on_next_pair(self):
        if not self.state.all_pairs:
            return
        n = len(self.state.all_pairs)
        self.state.pair_index = min(n - 1, self.state.pair_index + 1)
        self._render_pair()

    def _render_pair(self):
        pairs = self.state.all_pairs or []
        if not pairs:
            self.lbl_status.configure(text="No pairs loaded")
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
            return

        self._ensure_det_cache()
        i = self.state.pair_index
        n = len(pairs)
        self.btn_prev.config(state="normal" if i > 0 else "disabled")
        self.btn_next.config(state="normal" if i < n - 1 else "disabled")

        L, R = pairs[i]
        okL, cL, okR, cR = self.state.det_cache[i]
        both = okL and okR
        self.lbl_status.configure(
            text=f"Pair {i + 1}/{n} — left: {'OK' if okL else 'FAIL'} | "
                 f"right: {'OK' if okR else 'FAIL'} | both: {'OK' if both else 'FAIL'}"
        )
        self._draw_img(self.cnv_left, L, cL, okL)
        self._draw_img(self.cnv_right, R, cR, okR)

    def _draw_img(self, canvas: tk.Canvas, img_bgr, corners, ok):
        h, w = img_bgr.shape[:2]
        Hc, Wc = 360, 480
        scale = min(Wc / w, Hc / h)
        vis = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        if corners is not None:
            pts = (corners.reshape(-1, 2) * scale).astype(int)
            for x, y in pts:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        elif not ok:
            cv2.rectangle(vis, (0, 0), (int(w * scale), 28), (0, 0, 255), -1)
            cv2.putText(vis, "NOT DETECTED", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.image = imgtk
        canvas.delete("all")
        canvas.create_image(Wc // 2, Hc // 2, image=imgtk, anchor="center")


if __name__ == "__main__":
    app = StereoCalibGUI()
    app.geometry("1100x800")
    app.mainloop()
