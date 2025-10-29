
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from brillouin_system.eye_tracker.pupil_fitting.pupil_fitting_config.pupil_fit_config import PupilFitConfig, \
    left_eye_pupil_fit_config, right_eye_pupil_fit_config


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return img
    if img.shape[2] == 1: return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class EllipseFitter:
    """
    Holds left/right configs (from TOML via your config accessors) and provides:
      - find_pupil_left(image)
      - find_pupil_right(image)
    Both call a single internal algorithm _find_pupil(image, cfg).

    Configs are ALWAYS used (no ad-hoc overrides here). Call .refresh() to re-pull
    latest values if you edit the TOML or change them via your config GUI.
    """

    def __init__(self, cfg_left: PupilFitConfig = None, cfg_right: PupilFitConfig = None) -> None:

        if cfg_left is None:
            self.cfg_left = left_eye_pupil_fit_config.get()
        else:
            self.cfg_left = cfg_left
        if cfg_right is None:
            self.cfg_right = right_eye_pupil_fit_config.get()
        else:
            self.cfg_right = cfg_right


    def set_fitting_configs(self, cfg_left: PupilFitConfig, cfg_right: PupilFitConfig) -> None:
        self.cfg_left = cfg_left
        self.cfg_right = cfg_right

    # ---- Public API ----
    def find_pupil_left(self, image: np.ndarray) -> Ellipse2D:
        return self._find_pupil(image, self.cfg_left)

    def find_pupil_right(self, image: np.ndarray) -> Ellipse2D:
        return self._find_pupil(image, self.cfg_right)

    # ---- Core implementation shared by both cameras ----
    def _find_pupil(self, image: np.ndarray, cfg: PupilFitConfig) -> Ellipse2D | None:
        """
        Detect a dark pupil ellipse using the given cfg.
        Returns Ellipse2D in ORIGINAL image pixel coordinates, or None.
        """
        gray_full = _to_gray(image)
        H, W = gray_full.shape[:2]

        # ROI crop (if provided by config)
        if cfg.roi is not None:
            x, y, w, h = cfg.roi
            x = max(0, x); y = max(0, y)
            w = min(w, W - x); h = min(h, H - y)
            gray = gray_full[y:y+h, x:x+w]
            offx, offy = x, y
        else:
            gray = gray_full
            offx = offy = 0

        # Optional downscale for speed
        if 0.0 < cfg.scale < 1.0:
            small = cv2.resize(gray, None, fx=cfg.scale, fy=cfg.scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray

        # Light blur
        k = int(cfg.gaussian_ksize)
        if k % 2 == 0: k += 1
        k = max(k, 1)
        small_blur = cv2.GaussianBlur(small, (k, k), 0.0)

        # Threshold (dark pupil → invert)
        if getattr(cfg, "use_otsu", True):
            _, th = cv2.threshold(small_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        else:
            th = cv2.adaptiveThreshold(small_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)

        # Morphological close
        ck = int(cfg.close_kernel)
        ck = max(ck, 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, ker, iterations=int(cfg.close_iterations))

        # Contours & gating
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        img_area = float(small.shape[0] * small.shape[1])
        best = None
        best_area = -1.0

        for c in contours:
            if len(c) < 5:
                continue  # fitEllipse needs >=5 points
            area = cv2.contourArea(c)
            if area < cfg.min_area_frac * img_area or area > cfg.max_area_frac * img_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = max(w, h) / (min(w, h) + 1e-9)
            if ar > cfg.max_bbox_aspect:
                continue
            if area > best_area:
                best_area = area
                best = c

        if best is None:
            return None

        # Ellipse fit (in small coords)
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(best)

        # Rescale & shift back to original image coords
        if 0.0 < cfg.scale < 1.0:
            cx /= cfg.scale; cy /= cfg.scale
            MA /= cfg.scale; ma /= cfg.scale
        cx += offx; cy += offy

        # Ensure (major, minor) ordering
        if ma > MA:
            MA, ma = ma, MA
            ang = (ang + 90.0) % 180.0

        return Ellipse2D(cx=float(cx), cy=float(cy),
                         major=float(MA), minor=float(ma),
                         angle_deg=float(ang))


# Optional: quick helper to visualize a result
def draw_ellipse(image: np.ndarray, e: Optional[Ellipse2D],
                 color=(0, 255, 0), thickness=2) -> np.ndarray:
    if e is None:
        return image.copy()
    out = image.copy()
    center = (int(round(e.cx)), int(round(e.cy)))
    axes = (int(round(e.major / 2.0)), int(round(e.minor / 2.0)))
    cv2.ellipse(out, center, axes, e.angle_deg, 0, 360, color, thickness)
    cv2.circle(out, center, 2, color, -1)
    return out
