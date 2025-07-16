"""Low-level image utilities: load, auto-orient, detect paper contour, warp to
a top-down "scanned" view, and basic binarization.

These ops are intentionally conservative: if contour detection fails we return
the original image so the pipeline never breaks.

All functions operate on and return **PIL Images** externally; OpenCV (cv2) is
used internally on numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


@dataclass(frozen=True)
class Quad:
    """Four-point polygon in image pixel coords (tl, tr, br, bl)."""
    tl: Tuple[float, float]
    tr: Tuple[float, float]
    br: Tuple[float, float]
    bl: Tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.array([self.tl, self.tr, self.br, self.bl], dtype="float32")


def _order_points(pts: np.ndarray) -> Quad:
    """Return points ordered tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return Quad(tuple(tl), tuple(tr), tuple(br), tuple(bl))


def _largest_4pt_contour(edged: np.ndarray, min_area_ratio: float = 0.2) -> Optional[Quad]:
    """Find the largest contour that approximates to 4 points."""
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = edged.shape[0] * edged.shape[1]
    best = None
    best_area = 0

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < min_area_ratio * image_area:
            continue
        if area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None

    pts = best.reshape(4, 2)
    return _order_points(pts)


def load_pil(path: Path) -> Image.Image:
    """Load an image path into a PIL Image."""
    return Image.open(path)


def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL (RGB or L mode) -> OpenCV BGR/GRAY."""
    img = np.array(pil_img)
    if img.ndim == 2:
        return img
    # PIL gives RGB, cv2 expects BGR
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR/GRAY -> PIL RGB/L."""
    if cv_img.ndim == 2:
        return Image.fromarray(cv_img)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def auto_rotate_if_landscape(pil_img: Image.Image) -> Image.Image:
    """Invoices are portrait. If clearly landscape rotate 90°."""
    w, h = pil_img.size
    if w > h * 1.2:
        return pil_img.rotate(90, expand=True)
    return pil_img


def detect_and_warp(pil_img: Image.Image) -> Image.Image:
    """Detect paper contour and perspective-warp.

    Returns warped PIL image; if contour not found returns original.
    """
    pil_img = auto_rotate_if_landscape(pil_img)
    cv_img = pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    quad = _largest_4pt_contour(edged)
    if not quad:
        return pil_img

    # determine new size
    (tl, tr, br, bl) = quad.as_array()

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(quad.as_array(), dst)
    warped = cv2.warpPerspective(cv_img, M, (maxW, maxH))
    return cv_to_pil(warped)


def binarize_for_text(pil_img: Image.Image) -> Image.Image:
    """Contrast + adaptive threshold to sharpen text."""
    cv_img = pil_to_cv(pil_img)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    # Contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    # Adaptive threshold
    th = cv2.adaptiveThreshold(
        cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(th)


def crop_percent(pil_img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    """Crop a % box (x0, y0, x1, y1) in [0,1] coords."""
    w, h = pil_img.size
    x0, y0, x1, y1 = box
    return pil_img.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))


def expand_box(box: Tuple[float, float, float, float], pad: float = 0.01) -> Tuple[float, float, float, float]:
    """Expand a % box by pad each side (clamped)."""
    x0, y0, x1, y1 = box
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(1.0, x1 + pad)
    y1 = min(1.0, y1 + pad)
    return x0, y0, x1, y1
