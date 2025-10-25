
# point_capture_gui.py
from __future__ import annotations

import os
import threading
import csv
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ----------------- small helpers -----------------
def detect_dot_centroid(gray: np.ndarray, min_area: int) -> Optional[Tuple[float, float, float]]:
    """
    Detect a single black dot on white background.
    Returns (cx, cy, area) in pixel units or None if not found.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # binary inverse: dot is dark
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 3)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # keep contours above min_area
    cnts = [c for c in cnts if cv2.contourArea(c) >= float(min_area)]
    if not cnts:
        return None

    # choose largest blob
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < float(min_area):
        return None

    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # optional subpixel refine in a small window
    pt = np.array([[cx, cy]], np.float32)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    cv2.cornerSubPix(gray, pt, (5, 5), (-1, -1), term)
    return float(pt[0, 0]), float(pt[0, 1]), area


def load_image_pairs_simple(folder: str) -> List[Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Pair images by basename suffixes _left / _right (or -left / -right).
    Returns list of (img_left_bgr, img_right_bgr, path_left, path_right).
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    stems = {}
    for f in files:
        name, ext = os.path.splitext(f)
        if name.endswith("_left") or name.endswith("-left"):
            stem = name[:-5]
            stems.setdefault(stem, {})["left"] = os.path.join(folder, f)
        elif name.endswith("_right") or name.endswith("-right"):
            stem = name[:-6]
            stems.setdefault(stem, {})["right"] = os.path.join(folder, f)

    pairs = []
    for stem, sides in sorted(stems.items()):
        if "left" in sides and "right" in sides:
            L = cv2.imread(sides["left"], cv2.IMREAD_COLOR)
            R = cv2.imread(sides["right"], cv2.IMREAD_COLOR)
            if L is None or R is None:
                continue
            pairs.append((L, R, sides["left"], sides["right"]))
    return pairs


# ----------------- app state -----------------
@dataclass
class AppState:
    folder: str = ""
    model: str = "pinhole"   # keep camera parameter choice
    reference: str = "left"  # keep reference selector
    min_area: int = 200      # adjustable dot area threshold (px^2)

    pairs: Optional[List[Tuple[np.ndarray, np.ndarray, str, str]]] = None
    pair_index: int = 0

    # detection cache: per pair -> (okL, (cx,cy,area) or None, okR, (cx,cy,area) or None)
    det_cache: Optional[List[Tuple[bool, Optional[Tuple[float, float, float]],
                                   bool, Optional[Tuple[float, float, float]]]]] = None

    # user-entered coordinates per pair in micrometers (x, y, z) or None
    coords_um: Optional[List[Optional[Tuple[float, float, float]]]] = None


# ----------------- main GUI -----------------
class PointCaptureGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Point Capture GUI (Dot detection + per-pair coordinates)")
        self.state = AppState()
        self._build_ui()

    # ---------- UI ----------
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

        # Parameters row
        row += 1
        frame = ttk.Frame(root)
        frame.grid(row=row, column=0, columnspan=3, pady=10, sticky="we")

        self.var_model = tk.StringVar(value="pinhole")
        self.var_ref = tk.StringVar(value="left")
        self.var_min_area = tk.IntVar(value=200)

        ttk.Label(frame, text="Camera Model").grid(row=0, column=0, padx=(0, 4))
        ttk.Combobox(frame, textvariable=self.var_model, values=["pinhole", "fisheye"],
                    width=10, state="readonly").grid(row=0, column=1, padx=(0, 12))

        ttk.Label(frame, text="Reference").grid(row=0, column=2, padx=(0, 4))
        ttk.Combobox(frame, textvariable=self.var_ref, values=["left", "right"],
                    width=8, state="readonly").grid(row=0, column=3, padx=(0, 12))

        ttk.Label(frame, text="min_area (px²)").grid(row=0, column=4, padx=(0, 4))
        ttk.Entry(frame, textvariable=self.var_min_area, width=10).grid(row=0, column=5, padx=(0, 12))

        # Buttons
        row += 1
        bframe = ttk.Frame(root)
        bframe.grid(row=row, column=0, columnspan=3, pady=6)

        ttk.Button(bframe, text="Scan Frames", command=lambda: self._async(self._scan_frames_impl)).grid(row=0, column=0, padx=5)
        ttk.Button(bframe, text="Detect Current", command=self._detect_current_pair).grid(row=0, column=1, padx=5)
        ttk.Button(bframe, text="Detect All", command=lambda: self._async(self._detect_all_impl)).grid(row=0, column=2, padx=5)
        ttk.Button(bframe, text="Export (CSV)", command=lambda: self._export('csv')).grid(row=0, column=3, padx=5)
        ttk.Button(bframe, text="Export (JSON)", command=lambda: self._export('json')).grid(row=0, column=4, padx=5)
        ttk.Button(bframe, text="Export (TOML)", command=lambda: self._export('toml')).grid(row=0, column=5, padx=5)

        # Viewer controls
        row += 1
        self.viewer_frame = ttk.Frame(root)
        self.viewer_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=10)
        root.rowconfigure(row, weight=1)

        ctrl = ttk.Frame(self.viewer_frame)
        ctrl.grid(row=0, column=0, sticky="we", pady=(0, 6))
        self.btn_prev = ttk.Button(ctrl, text="⟵ Prev", command=self._on_prev_pair, state="disabled")
        self.btn_next = ttk.Button(ctrl, text="Next ⟶", command=self._on_next_pair, state="disabled")
        self.lbl_status = ttk.Label(ctrl, text="No pairs loaded")
        self.lbl_detect = ttk.Label(ctrl, text="• Idle", foreground="#999")
        self.btn_prev.grid(row=0, column=0, padx=5)
        self.btn_next.grid(row=0, column=1, padx=5)
        self.lbl_status.grid(row=0, column=2, padx=10)
        self.lbl_detect.grid(row=0, column=3, padx=10)
        ctrl.columnconfigure(2, weight=1)

        canv = ttk.Frame(self.viewer_frame)
        canv.grid(row=1, column=0, sticky="nsew")
        self.cnv_left = tk.Canvas(canv, width=480, height=360, bg="#111")
        self.cnv_right = tk.Canvas(canv, width=480, height=360, bg="#111")
        self.cnv_left.grid(row=0, column=0, padx=4, pady=4)
        self.cnv_right.grid(row=0, column=1, padx=4, pady=4)
        canv.columnconfigure(0, weight=1)
        canv.columnconfigure(1, weight=1)

        # Coordinate entry panel (micrometers)
        coord = ttk.Frame(self.viewer_frame)
        coord.grid(row=2, column=0, sticky="we", pady=(8, 0))
        ttk.Label(coord, text="Coordinates (µm):").grid(row=0, column=0, sticky="w")
        self.var_x = tk.StringVar()
        self.var_y = tk.StringVar()
        self.var_z = tk.StringVar()
        ttk.Label(coord, text="x").grid(row=0, column=1); ttk.Entry(coord, textvariable=self.var_x, width=10).grid(row=0, column=2, padx=4)
        ttk.Label(coord, text="y").grid(row=0, column=3); ttk.Entry(coord, textvariable=self.var_y, width=10).grid(row=0, column=4, padx=4)
        ttk.Label(coord, text="z").grid(row=0, column=5); ttk.Entry(coord, textvariable=self.var_z, width=10).grid(row=0, column=6, padx=4)
        ttk.Button(coord, text="Save coords for this pair", command=self._save_coords_for_pair).grid(row=0, column=7, padx=10)

        # Log
        row += 1
        ttk.Label(root, text="Log:").grid(row=row, column=0, sticky="w")
        row += 1
        self.txt_log = tk.Text(root, height=8)
        self.txt_log.grid(row=row, column=0, columnspan=3, sticky="nsew")
        root.rowconfigure(row, weight=1)

    # ---------- helpers ----------
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

    def _flash_detect(self, ok_left: bool, ok_right: bool):
        if ok_left and ok_right:
            self.lbl_detect.configure(text="• Detected (both)", foreground="#16a34a")  # green
        elif ok_left or ok_right:
            self.lbl_detect.configure(text="• Partial detection", foreground="#ca8a04")  # amber
        else:
            self.lbl_detect.configure(text="• No detection", foreground="#dc2626")  # red

    # ---------- operations ----------
    def _scan_frames_impl(self):
        folder = self.var_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            self._log("Invalid folder")
            return
        self._log("Scanning for left/right image pairs…")
        pairs = load_image_pairs_simple(folder)
        if not pairs:
            self._log("No pairs found (expect *_left.* and *_right.*)")
            self.state.pairs = None
            self.state.det_cache = None
            self.state.coords_um = None
            self.state.pair_index = 0
            self._render_pair()
            return
        self.state.pairs = pairs
        self._log(f"Found {len(pairs)} pairs")
        self.state.det_cache = [(False, None, False, None) for _ in pairs]
        self.state.coords_um = [None for _ in pairs]
        self.state.pair_index = 0
        self._render_pair()

    def _detect_current_pair(self):
        if not self.state.pairs:
            self._log("No pairs loaded")
            self._flash_detect(False, False)
            return
        i = self.state.pair_index
        L, R, _, _ = self.state.pairs[i]
        min_area = int(self.var_min_area.get())
        resL = detect_dot_centroid(L, min_area)
        resR = detect_dot_centroid(R, min_area)
        okL = resL is not None
        okR = resR is not None
        self.state.det_cache[i] = (okL, resL, okR, resR)
        self._flash_detect(okL, okR)
        self._log(f"Pair {i+1}: left={'OK' if okL else 'FAIL'} right={'OK' if okR else 'FAIL'} (min_area={min_area})")
        self._render_pair()

    def _detect_all_impl(self):
        if not self.state.pairs:
            self._log("No pairs loaded")
            self._flash_detect(False, False)
            return
        min_area = int(self.var_min_area.get())
        self._log(f"Detecting dots on all pairs (min_area={min_area})…")
        cache = []
        for i, (L, R, _, _) in enumerate(self.state.pairs, 1):
            resL = detect_dot_centroid(L, min_area)
            resR = detect_dot_centroid(R, min_area)
            cache.append((resL is not None, resL, resR is not None, resR))
            if i % 10 == 0:
                self._log(f"  processed {i}/{len(self.state.pairs)}")
        self.state.det_cache = cache
        # summarize
        ok_both = sum(1 for okL, _, okR, _ in cache if okL and okR)
        self._log(f"Detection complete. Both-sides detections: {ok_both}/{len(cache)}")
        # flash last pair status
        if cache:
            self._flash_detect(cache[-1][0], cache[-1][2])
        self._render_pair()

    def _save_coords_for_pair(self):
        if not self.state.pairs:
            self._log("No pairs loaded")
            return
        i = self.state.pair_index
        # parse floats (micrometers)
        try:
            x = float(self.var_x.get())
            y = float(self.var_y.get())
            z = float(self.var_z.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric x, y, z in micrometers.")
            return
        self.state.coords_um[i] = (x, y, z)
        self._log(f"Saved coords for pair {i+1}: ({x:.3f}, {y:.3f}, {z:.3f}) µm")

    def _export(self, fmt: str):
        if not self.state.pairs or not self.state.det_cache:
            self._log("Nothing to export")
            return
        # ask path
        if fmt == 'csv':
            out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="dot_detections.csv")
        elif fmt == 'json':
            out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="dot_detections.json")
        elif fmt == 'toml':
            out = filedialog.asksaveasfilename(defaultextension=".toml", filetypes=[("TOML", "*.toml")], initialfile="dot_detections.toml")
        else:
            return
        if not out:
            return

        data_rows = []
        for i, ((L, R, pL, pR), (okL, cL, okR, cR), coords) in enumerate(zip(self.state.pairs, self.state.det_cache, self.state.coords_um or []), 1):
            row = {
                "index": i,
                "left_path": pL,
                "right_path": pR,
                "left_cx": None if not cL else float(cL[0]),
                "left_cy": None if not cL else float(cL[1]),
                "left_area": None if not cL else float(cL[2]),
                "right_cx": None if not cR else float(cR[0]),
                "right_cy": None if not cR else float(cR[1]),
                "right_area": None if not cR else float(cR[2]),
                "detected_both": bool(okL and okR),
                "camera_model": self.var_model.get(),
                "reference": self.var_ref.get(),
                "min_area": int(self.var_min_area.get()),
                "coords_um": None if coords is None else {"x": coords[0], "y": coords[1], "z": coords[2]},
            }
            data_rows.append(row)

        try:
            if fmt == 'csv':
                with open(out, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["index", "left_path", "right_path",
                                "left_cx", "left_cy", "left_area",
                                "right_cx", "right_cy", "right_area",
                                "detected_both", "camera_model", "reference", "min_area",
                                "coords_um_x", "coords_um_y", "coords_um_z"])
                    for r in data_rows:
                        cu = r["coords_um"] or {}
                        w.writerow([
                            r["index"], r["left_path"], r["right_path"],
                            r["left_cx"], r["left_cy"], r["left_area"],
                            r["right_cx"], r["right_cy"], r["right_area"],
                            r["detected_both"], r["camera_model"], r["reference"], r["min_area"],
                            cu.get("x", ""), cu.get("y", ""), cu.get("z", ""),
                        ])
            elif fmt == 'json':
                with open(out, "w", encoding="utf-8") as f:
                    json.dump({"rows": data_rows}, f, indent=2)
            elif fmt == 'toml':
                # Minimal TOML writer (no extra dependency)
                def toml_escape(s: str) -> str:
                    return s.replace('\\', '\\\\').replace('"', '\"')
                with open(out, "w", encoding="utf-8") as f:
                    f.write('# dot detections export\n')
                    f.write(f'model = "{toml_escape(self.var_model.get())}"\n')
                    f.write(f'reference = "{toml_escape(self.var_ref.get())}"\n')
                    f.write(f'min_area = {int(self.var_min_area.get())}\n')
                    f.write("\n[[rows]]\n")
                    for idx, r in enumerate(data_rows):
                        if idx > 0:
                            f.write("\n[[rows]]\n")
                        f.write(f'index = {r["index"]}\n')
                        f.write(f'left_path = "{toml_escape(r["left_path"])}"\n')
                        f.write(f'right_path = "{toml_escape(r["right_path"])}"\n')
                        for key in ("left_cx","left_cy","left_area","right_cx","right_cy","right_area"):
                            val = r[key]
                            if val is None:
                                f.write(f'{key} = null\n')
                            else:
                                f.write(f'{key} = {float(val)}\n')
                        f.write(f'detected_both = {"true" if r["detected_both"] else "false"}\n')
                        cu = r["coords_um"]
                        if cu is None:
                            f.write('coords_um = { x = null, y = null, z = null }\n')
                        else:
                            f.write(f'coords_um = {{ x = {cu["x"]}, y = {cu["y"]}, z = {cu["z"]} }}\n')
            self._log(f"Saved {fmt.upper()} to: {out}")
            messagebox.showinfo("Export", f"Saved {fmt.UPPER()} to:\n{out}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ---------- viewer ----------
    def _on_prev_pair(self):
        if not self.state.pairs:
            return
        self.state.pair_index = max(0, self.state.pair_index - 1)
        self._render_pair()

    def _on_next_pair(self):
        if not self.state.pairs:
            return
        n = len(self.state.pairs)
        self.state.pair_index = min(n - 1, self.state.pair_index + 1)
        self._render_pair()

    def _render_pair(self):
        pairs = self.state.pairs or []
        if not pairs:
            self.lbl_status.configure(text="No pairs loaded")
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
            self._draw_blank(self.cnv_left)
            self._draw_blank(self.cnv_right)
            return

        i = self.state.pair_index
        n = len(pairs)
        self.btn_prev.config(state="normal" if i > 0 else "disabled")
        self.btn_next.config(state="normal" if i < n - 1 else "disabled")

        L, R, pL, pR = pairs[i]
        okL, cL, okR, cR = self.state.det_cache[i] if self.state.det_cache and i < len(self.state.det_cache) else (False, None, False, None)
        both = okL and okR
        self.lbl_status.configure(
            text=f"Pair {i+1}/{n}  |  left: {'OK' if okL else 'FAIL'}  |  right: {'OK' if okR else 'FAIL'}  |  both: {'OK' if both else 'FAIL'}"
        )
        # Load coords into entry boxes if saved
        if self.state.coords_um and self.state.coords_um[i] is not None:
            x, y, z = self.state.coords_um[i]
            self.var_x.set(f"{x}")
            self.var_y.set(f"{y}")
            self.var_z.set(f"{z}")
        else:
            self.var_x.set(self.var_x.get() if self.var_x.get() else "")
            self.var_y.set(self.var_y.get() if self.var_y.get() else "")
            self.var_z.set(self.var_z.get() if self.var_z.get() else "")

        self._draw_img(self.cnv_left, L, cL)
        self._draw_img(self.cnv_right, R, cR)

    def _draw_blank(self, canvas: tk.Canvas):
        canvas.delete("all")
        Wc, Hc = 480, 360
        canvas.create_rectangle(0, 0, Wc, Hc, fill="#111", outline="#111")

    def _draw_img(self, canvas: tk.Canvas, img_bgr, dot_info):
        h, w = img_bgr.shape[:2]
        Hc, Wc = 360, 480
        scale = min(Wc / w, Hc / h)
        vis = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        if dot_info is not None:
            cx, cy, area = dot_info
            x = int(cx * scale)
            y = int(cy * scale)
            cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
            cv2.putText(vis, f"({cx:.1f},{cy:.1f}) a={area:.0f}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showerror("Error", "Pillow is required (pip install pillow)")
            return
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.image = imgtk
        canvas.delete("all")
        canvas.create_image(Wc // 2, Hc // 2, image=imgtk, anchor="center")


if __name__ == "__main__":
    app = PointCaptureGUI()
    app.geometry("1150x850")
    app.mainloop()
