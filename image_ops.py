# image_ops.py
"""
Basic image operations for Invoice OCR:
  – label detection via EasyOCR
  – ROI cropping by percentage
  – box expansion
  – page warping / deskewing (stubbed)
  – binarization for Tesseract
"""

from typing import Dict, List, Tuple
from PIL import Image, ImageOps
import easyocr


# --------------------------------------------------------------------------- #
#                           LABEL SYNONYMS                                     #
# --------------------------------------------------------------------------- #
LABELS: Dict[str, List[str]] = {
    "invoice_number": ["Numéro", "Numero", "Num", "NUMÉRO"],
    "invoice_date":   ["Date", "date"],
    "mode_reglement": ["Mode de règlement", "Mode de reglement", "mode reglement"],
    "reference":      ["Référence", "Reference", "REFÉRENCE"],
    "mf":             ["M/F", "MF"],
    "ccb":            ["C.C.B", "CCB", "RIB"],
    "total_ht":       ["Total HT", "TotalHt", "Total Ht"],
}


def detect_and_warp(img: Image.Image) -> Image.Image:
    """
    Stub for page‐warping / deskewing.
    Currently returns the image unmodified.
    (Replace with perspective‐transform logic if needed.)
    """
    return img


def binarize_for_text(img: Image.Image) -> Image.Image:
    """
    Convert to clean black‑and‑white for Tesseract.
    Uses a fixed threshold on the L channel.
    """
    gray = img.convert("L")
    # simple 128 threshold → B/W
    bw = gray.point(lambda x: 0 if x < 128 else 255, mode="1")
    return bw.convert("RGB")  # return 3‑channel for consistency


def detect_label_rects(
    img: Image.Image
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Run EasyOCR on `img` to find any of the LABELS.
    Returns a dict mapping the *exact* matched label text → its bounding box (l, t, r, b).
    """
    reader = easyocr.Reader(["fr", "en"], gpu=False)
    results = reader.readtext(img, detail=1)
    rects: Dict[str, Tuple[int, int, int, int]] = {}

    for bbox, text, _prob in results:
        cleaned = text.strip()
        for roi_key, synonyms in LABELS.items():
            if any(cleaned.lower() == syn.lower() for syn in synonyms):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                l, t, r, b = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                rects[cleaned] = (l, t, r, b)
                break

    return rects


def find_label_box(
    label_rects: Dict[str, Tuple[int, int, int, int]],
    labels: List[str],
) -> Tuple[int, int, int, int] | None:
    """
    Given the dict from detect_label_rects (keys are matched texts),
    and a list of synonyms for one ROI field, return the first bbox found.
    """
    for syn in labels:
        for found_text, rect in label_rects.items():
            if found_text.lower() == syn.lower():
                return rect
    return None


def crop_percent(
    img: Image.Image,
    pct_box: Tuple[float, float, float, float],
) -> Image.Image:
    """
    Crop `img` by relative percentages (l_pct, t_pct, r_pct, b_pct).
    Returns a new Image.
    """
    l_pct, t_pct, r_pct, b_pct = pct_box
    w, h = img.width, img.height
    l, t, r, b = int(l_pct * w), int(t_pct * h), int(r_pct * w), int(b_pct * h)
    return img.crop((l, t, r, b))


def expand_box(
    pct_box: Tuple[float, float, float, float],
    pad: float = 0.01
) -> Tuple[float, float, float, float]:
    """
    Expand a (l, t, r, b) pct_box by `pad` in all directions (clamped 0–1).
    """
    l, t, r, b = pct_box
    return (
        max(0.0, l - pad),
        max(0.0, t - pad),
        min(1.0, r + pad),
        min(1.0, b + pad),
    )
