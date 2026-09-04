# stereo_calib_gui.py
from __future__ import annotations

import os
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

from brillouin_system.eye_tracker.stereo_imaging.calibrate_single import (
    MonoCalibConfig,
    CameraResult,
    calibrate_single,
    save_camera_json,
    detect_corners,
)
from brillouin_system.eye_tracker.stereo_imaging.calibrate_stereo import (
    StereoCalibConfig,
    PairsValid,
    load_image_pairs_smart,
    stereo_calibrate_from_pairs,
    save_stereo_json,
)


# ---------------- per-pair record ----------------
@dataclass
class PairRecord:
    index: int
    L: np.ndarray
    R: np.ndarray
    cL: Optional[np.ndarray] = None      # cached corner detection (left)
    cR: Optional[np.ndarray] = None      # cached corner detection (right)
    use: bool = True                     # include this pair in calibration
    mL: Optional[Dict] = None            # quality metrics (left)
    mR: Optional[Dict] = None            # quality metrics (right)
    errL: Optional[float] = None         # per-view reproj RMS after left mono
    errR: Optional[float] = None         # per-view reproj RMS after right mono

    @property
    def okL(self) -> bool:
        return self.cL is not None

    @property
    def okR(self) -> bool:
        return self.cR is not None


# ---------------- quality metrics & hints ----------------
_CELL_NAMES = {
    (0, 0): "top-left", (1, 0): "top-center", (2, 0): "top-right",
    (0, 1): "middle-left", (1, 1): "center", (2, 1): "middle-right",
    (0, 2): "bottom-left", (1, 2): "bottom-center", (2, 2): "bottom-right",
}

TILT_GOOD = 0.12          # foreshortening score above which a view counts as "tilted"
ERR_WARN_PX = 1.0         # per-view mono reprojection error worth flagging
STEREO_RMS_WARN_PX = 1.0  # stereo RMS above which we warn


def corner_metrics(corners: np.ndarray, img_shape, pattern_size) -> Dict:
    """Board coverage/pose metrics for one detection."""
    h, w = img_shape[:2]
    pts = corners.reshape(-1, 2).astype(np.float64)
    hull = cv2.convexHull(pts.astype(np.float32))
    area_frac = float(cv2.contourArea(hull)) / float(w * h)

    cx, cy = pts.mean(axis=0)
    cell = (min(2, int(3 * cx / w)), min(2, int(3 * cy / h)))

    cols, rows = int(pattern_size[0]), int(pattern_size[1])
    p00 = pts[0]
    p01 = pts[cols - 1]
    p10 = pts[(rows - 1) * cols]
    p11 = pts[-1]
    top = np.linalg.norm(p01 - p00)
    bottom = np.linalg.norm(p11 - p10)
    left = np.linalg.norm(p10 - p00)
    right = np.linalg.norm(p11 - p01)
    r_h = min(top, bottom) / max(top, bottom) if max(top, bottom) > 0 else 1.0
    r_v = min(left, right) / max(left, right) if max(left, right) > 0 else 1.0
    tilt = 1.0 - min(r_h, r_v)  # 0 = fronto-parallel, larger = more oblique

    return {"area_frac": area_frac, "cell": cell, "tilt": tilt}


def coverage_hints(name: str, metrics: List[Dict]) -> List[str]:
    """Human-readable advice: what is covered, what is missing."""
    if not metrics:
        return [f"{name}: no usable detections."]

    hints: List[str] = []
    n = len(metrics)
    if n < 12:
        hints.append(f"{name}: only {n} usable views — aim for 15–30.")

    covered = {m["cell"] for m in metrics}
    missing = [_CELL_NAMES[c] for c in sorted(_CELL_NAMES) if c not in covered]
    if missing:
        hints.append(f"{name}: board never centered in: {', '.join(missing)}.")

    n_tilted = sum(1 for m in metrics if m["tilt"] >= TILT_GOOD)
    if n_tilted < 5:
        hints.append(
            f"{name}: only {n_tilted} tilted views — add oblique poses "
            f"(~20–40° tilt); they pin down focal length and distortion."
        )

    areas = [m["area_frac"] for m in metrics]
    if max(areas) < 0.2:
        hints.append(f"{name}: board always small (max {max(areas)*100:.0f}% of image) — add close-ups.")
    if min(areas) > 0 and max(areas) / max(min(areas), 1e-9) < 2.5:
        hints.append(f"{name}: little distance variation — capture both near and far views.")

    if not hints:
        hints.append(f"{name}: coverage looks good ({n} views, all 9 image regions, "
                     f"{n_tilted} tilted).")
    return hints


# ---------------- GUI ----------------
class StereoCalibGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stereo Calibration GUI")

        self.records: List[PairRecord] = []
        self.left_res: Optional[CameraResult] = None
        self.right_res: Optional[CameraResult] = None
        self.pair_index: int = 0

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

        # Parameters (defaults match the capture GUI: 10x8 inner corners)
        row += 1
        frame = ttk.Frame(root)
        frame.grid(row=row, column=0, columnspan=3, pady=10, sticky="we")
        self.var_cols = tk.IntVar(value=10)
        self.var_rows = tk.IntVar(value=8)
        self.var_size = tk.DoubleVar(value=2.0)
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
        ttk.Button(bframe, text="Coverage Hints", command=self._log_hints).grid(row=0, column=4, padx=5)

        # Pair table
        row += 1
        tframe = ttk.Frame(root)
        tframe.grid(row=row, column=0, columnspan=3, sticky="nsew")
        root.rowconfigure(row, weight=1)

        cols = ("use", "left", "right", "areaL", "areaR", "tilt", "errL", "errR")
        self.tree = ttk.Treeview(tframe, columns=cols, show="tree headings", height=9, selectmode="browse")
        self.tree.heading("#0", text="Pair")
        self.tree.column("#0", width=70, anchor="w")
        headings = {
            "use": ("Use", 45), "left": ("L det", 55), "right": ("R det", 55),
            "areaL": ("L area%", 65), "areaR": ("R area%", 65),
            "tilt": ("Tilt%", 55), "errL": ("L err px", 65), "errR": ("R err px", 65),
        }
        for c, (label, width) in headings.items():
            self.tree.heading(c, text=label)
            self.tree.column(c, width=width, anchor="center")

        scroll = ttk.Scrollbar(tframe, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        tframe.rowconfigure(0, weight=1)
        tframe.columnconfigure(0, weight=1)

        self.tree.tag_configure("excluded", foreground="#999999")
        self.tree.tag_configure("bad", foreground="#cc0000")

        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<space>", self._on_tree_space)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Viewer
        row += 1
        self.viewer_frame = ttk.Frame(root)
        self.viewer_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=8)

        ctrl = ttk.Frame(self.viewer_frame)
        ctrl.grid(row=0, column=0, sticky="we", pady=(0, 6))
        self.btn_prev = ttk.Button(ctrl, text="⟵ Prev", command=self._on_prev_pair, state="disabled")
        self.btn_next = ttk.Button(ctrl, text="Next ⟶", command=self._on_next_pair, state="disabled")
        self.btn_toggle = ttk.Button(ctrl, text="Include/Exclude (Space)", command=self._toggle_current, state="disabled")
        self.lbl_status = ttk.Label(ctrl, text="No pairs loaded")
        self.btn_prev.grid(row=0, column=0, padx=5)
        self.btn_next.grid(row=0, column=1, padx=5)
        self.btn_toggle.grid(row=0, column=2, padx=5)
        self.lbl_status.grid(row=0, column=3, padx=10)
        ctrl.columnconfigure(3, weight=1)

        canv = ttk.Frame(self.viewer_frame)
        canv.grid(row=1, column=0, sticky="nsew")
        self.cnv_left = tk.Canvas(canv, width=480, height=330, bg="#111")
        self.cnv_right = tk.Canvas(canv, width=480, height=330, bg="#111")
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

    # ---------------- Core helpers ----------------
    def _on_browse(self):
        folder = filedialog.askdirectory()
        if folder:
            self.var_folder.set(folder)

    def _async(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _log(self, msg: str):
        def _do():
            self.txt_log.insert("end", msg + "\n")
            self.txt_log.see("end")
        self.after(0, _do)

    def _pattern(self) -> Tuple[int, int]:
        return int(self.var_cols.get()), int(self.var_rows.get())

    def _used(self) -> List[PairRecord]:
        return [r for r in self.records if r.use]

    # ---------------- pair table ----------------
    def _refresh_tree(self):
        def _fmt(v, scale=1.0, digits=1):
            return f"{v * scale:.{digits}f}" if v is not None else "–"

        def _do():
            self.tree.delete(*self.tree.get_children())
            for r in self.records:
                tags = []
                if not r.use:
                    tags.append("excluded")
                elif ((r.errL is not None and r.errL > ERR_WARN_PX)
                      or (r.errR is not None and r.errR > ERR_WARN_PX)
                      or not (r.okL and r.okR)):
                    tags.append("bad")
                tilt = max(
                    r.mL["tilt"] if r.mL else 0.0,
                    r.mR["tilt"] if r.mR else 0.0,
                ) if (r.mL or r.mR) else None
                self.tree.insert(
                    "", "end", iid=str(r.index), text=f"#{r.index:04d}",
                    values=(
                        "☑" if r.use else "☐",
                        "OK" if r.okL else "FAIL",
                        "OK" if r.okR else "FAIL",
                        _fmt(r.mL["area_frac"] if r.mL else None, 100.0),
                        _fmt(r.mR["area_frac"] if r.mR else None, 100.0),
                        _fmt(tilt, 100.0, 0),
                        _fmt(r.errL, digits=2),
                        _fmt(r.errR, digits=2),
                    ),
                    tags=tags,
                )
            if self.records:
                idx = min(self.pair_index, len(self.records) - 1)
                iid = str(self.records[idx].index)
                self.tree.selection_set(iid)
                self.tree.see(iid)
        self.after(0, _do)

    def _on_tree_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        if self.tree.identify_column(event.x) == "#1":  # "use" column
            self._toggle_record(int(iid))
            return "break"

    def _on_tree_space(self, event):
        sel = self.tree.selection()
        if sel:
            self._toggle_record(int(sel[0]))
        return "break"

    def _on_tree_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        for i, r in enumerate(self.records):
            if r.index == idx:
                if self.pair_index != i:
                    self.pair_index = i
                    self._render_pair()
                break

    def _toggle_record(self, index: int):
        for r in self.records:
            if r.index == index:
                r.use = not r.use
                break
        self._refresh_tree()
        self._render_pair()

    def _toggle_current(self):
        if self.records:
            self._toggle_record(self.records[self.pair_index].index)

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

        cols, rows = self._pattern()
        pattern = (cols, rows)
        records: List[PairRecord] = []
        n_okL = n_okR = n_both = 0
        for i, (L, R) in enumerate(pairs):
            cL = detect_corners(L, pattern)
            cR = detect_corners(R, pattern)
            rec = PairRecord(index=i, L=L, R=R, cL=cL, cR=cR)
            if cL is not None:
                rec.mL = corner_metrics(cL, L.shape, pattern)
                n_okL += 1
            if cR is not None:
                rec.mR = corner_metrics(cR, R.shape, pattern)
                n_okR += 1
            if cL is not None and cR is not None:
                n_both += 1
            else:
                rec.use = cL is not None or cR is not None  # keep for mono if one side works
            records.append(rec)
            self._log(f"  pair {i:04d}: left {'OK' if cL is not None else 'FAIL'} | right {'OK' if cR is not None else 'FAIL'}")

        self.records = records
        self.left_res = None
        self.right_res = None
        self.pair_index = 0
        self._log(f"Valid left: {n_okL} | right: {n_okR} | pairs (both): {n_both} of {len(pairs)}")
        self._refresh_tree()
        self._render_pair()
        self._log_hints()

    def _log_hints(self):
        if not self.records:
            self._log("Scan first — no pairs loaded.")
            return
        used = self._used()
        mL = [r.mL for r in used if r.mL is not None]
        mR = [r.mR for r in used if r.mR is not None]
        self._log("--- Coverage hints (included frames only) ---")
        for line in coverage_hints("LEFT", mL) + coverage_hints("RIGHT", mR):
            self._log("  " + line)
        n_both = sum(1 for r in used if r.okL and r.okR)
        if n_both < 10:
            self._log(f"  STEREO: only {n_both} included pairs with both sides detected — aim for 10+.")

    def _run_left_mono_impl(self):
        self._run_mono_impl("left")

    def _run_right_mono_impl(self):
        self._run_mono_impl("right")

    def _run_mono_impl(self, side: str):
        recs = [r for r in self._used() if (r.okL if side == "left" else r.okR)]
        if not recs:
            self._log(f"No included {side} frames with detections. Scan first.")
            return
        cfg = MonoCalibConfig(model=self.var_model.get(), cols=int(self.var_cols.get()),
                              rows=int(self.var_rows.get()), square_size_mm=float(self.var_size.get()))
        self._log(f"Running {side} mono calibration on {len(recs)} frames…")
        images = [(r.L if side == "left" else r.R) for r in recs]
        corners = [(r.cL if side == "left" else r.cR) for r in recs]
        try:
            res, imgsize = calibrate_single(images, cfg, corners=corners)
        except Exception as e:
            self._log(f"{side} mono calibration failed: {e}")
            return

        # Map per-view errors back to the pair table
        if res.per_view_rms is not None and len(res.per_view_rms) == len(recs):
            for r, err in zip(recs, res.per_view_rms):
                if side == "left":
                    r.errL = float(err)
                else:
                    r.errR = float(err)
            worst = sorted(recs, key=lambda r: (r.errL if side == "left" else r.errR), reverse=True)[:3]
            worst_txt = ", ".join(
                f"#{r.index:04d} ({(r.errL if side == 'left' else r.errR):.2f} px)" for r in worst
            )
            self._log(f"{side} mono RMS = {res.rms:.3f} px | worst frames: {worst_txt}")
            if any((r.errL if side == "left" else r.errR) > ERR_WARN_PX for r in worst):
                self._log(f"  Hint: frames above {ERR_WARN_PX:.1f} px are marked red — "
                          f"consider excluding them (click Use / Space) and re-running.")
        else:
            self._log(f"{side} mono RMS = {res.rms:.3f} px")

        if side == "left":
            self.left_res = res
        else:
            self.right_res = res
        out = os.path.join(self.var_folder.get(), f"{self.var_prefix.get()}_{side}.json")
        save_camera_json(out, res, imgsize, cfg)
        self._log(f"Saved {out}")
        self._refresh_tree()

    def _run_stereo_impl(self):
        recs = [r for r in self._used() if r.okL and r.okR]
        if not recs:
            self._log("No included pairs with both sides detected.")
            return
        if not self.left_res or not self.right_res:
            self._log("Run both mono calibrations first!")
            return
        cfg = StereoCalibConfig(model=self.var_model.get(), reference=self.var_ref.get(),
                                cols=int(self.var_cols.get()), rows=int(self.var_rows.get()),
                                square_size_mm=float(self.var_size.get()))
        self._log(f"Running stereo calibration (extrinsics only) on {len(recs)} pairs…")
        try:
            stereo = stereo_calibrate_from_pairs(
                PairsValid([(r.L, r.R) for r in recs]), cfg,
                self.left_res, self.right_res,
                corners=[(r.cL, r.cR) for r in recs],
            )
        except Exception as e:
            self._log(f"Stereo calibration failed: {e}")
            return

        baseline = float(np.linalg.norm(stereo.T))
        self._log(f"Stereo RMS = {stereo.rms:.3f} px | baseline = {baseline:.2f} mm "
                  f"(check this against the real camera separation!)")
        if stereo.rms > STEREO_RMS_WARN_PX:
            self._log(f"  Warning: stereo RMS above {STEREO_RMS_WARN_PX:.1f} px — "
                      f"exclude outlier pairs (red rows) and re-run monos + stereo.")

        h, w = recs[0].L.shape[:2]
        out = os.path.join(self.var_folder.get(), f"{self.var_prefix.get()}_stereo.json")
        save_stereo_json(out, stereo, cfg, (w, h))
        self._log(f"Saved {out}")
        self.after(0, lambda: messagebox.showinfo(
            "Done", f"Stereo calibration complete!\nRMS = {stereo.rms:.3f} px, baseline = {baseline:.2f} mm"))

    # ---------------- Viewer ----------------
    def _on_prev_pair(self):
        if not self.records:
            return
        self.pair_index = max(0, self.pair_index - 1)
        self._render_pair()
        self._sync_tree_selection()

    def _on_next_pair(self):
        if not self.records:
            return
        self.pair_index = min(len(self.records) - 1, self.pair_index + 1)
        self._render_pair()
        self._sync_tree_selection()

    def _sync_tree_selection(self):
        if self.records:
            iid = str(self.records[self.pair_index].index)
            self.tree.selection_set(iid)
            self.tree.see(iid)

    def _render_pair(self):
        def _do():
            if not self.records:
                self.lbl_status.configure(text="No pairs loaded")
                self.btn_prev.config(state="disabled")
                self.btn_next.config(state="disabled")
                self.btn_toggle.config(state="disabled")
                return

            i = self.pair_index
            n = len(self.records)
            r = self.records[i]
            self.btn_prev.config(state="normal" if i > 0 else "disabled")
            self.btn_next.config(state="normal" if i < n - 1 else "disabled")
            self.btn_toggle.config(state="normal")

            both = r.okL and r.okR
            self.lbl_status.configure(
                text=f"Pair #{r.index:04d} ({i + 1}/{n}) — left: {'OK' if r.okL else 'FAIL'} | "
                     f"right: {'OK' if r.okR else 'FAIL'} | both: {'OK' if both else 'FAIL'} | "
                     f"{'INCLUDED' if r.use else 'EXCLUDED'}"
            )
            self._draw_img(self.cnv_left, r.L, r.cL, r.okL, r.use)
            self._draw_img(self.cnv_right, r.R, r.cR, r.okR, r.use)
        self.after(0, _do)

    def _draw_img(self, canvas: tk.Canvas, img_bgr, corners, ok, use):
        h, w = img_bgr.shape[:2]
        Hc, Wc = 330, 480
        scale = min(Wc / w, Hc / h)
        vis = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        if corners is not None:
            pts = (corners.reshape(-1, 2) * scale).astype(int)
            for x, y in pts:
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        elif not ok:
            cv2.rectangle(vis, (0, 0), (int(w * scale), 28), (0, 0, 255), -1)
            cv2.putText(vis, "NOT DETECTED", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if not use:
            cv2.rectangle(vis, (0, int(h * scale) - 28), (int(w * scale), int(h * scale)), (128, 128, 128), -1)
            cv2.putText(vis, "EXCLUDED", (8, int(h * scale) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.image = imgtk
        canvas.delete("all")
        canvas.create_image(Wc // 2, Hc // 2, image=imgtk, anchor="center")


if __name__ == "__main__":
    app = StereoCalibGUI()
    app.geometry("1100x950")
    app.mainloop()
