# ocr_service.py
"""
OCR service – EasyOCR + Tesseract pipeline tailored for mixed
French/English TunTrust invoices.

Main entry point
----------------
run_structured_ocr(path: pathlib.Path) → tuple[str, dict[str, str]]

Returns
• full_text – raw EasyOCR text for the whole page
• roi_texts – keyed result for each region of interest (invoice_number, invoice_date, …, totals_table)
"""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import Dict, Tuple

import easyocr
from PIL import Image

from . import image_ops as iops


# --------------------------------------------------------------------------- #
#                         CONSTANTS / LOOK‑UP TABLES                          #
# --------------------------------------------------------------------------- #

_ROI_TESS_CFG: Dict[str, Tuple[str, str]] = {
    "invoice_number": ("eng+fra", "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/- "),
    "invoice_date":   ("eng+fra", "--psm 7 -c tessedit_char_whitelist=0123456789/- "),
    "mode_reglement": ("fra",     "--psm 7"),
    "reference":      ("fra",     "--psm 7"),
    "mf":             ("eng+fra", "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "),
    "ccb":            ("eng",     "--psm 7 -c tessedit_char_whitelist=0123456789 "),
    "lines_table":    ("fra",     "--psm 6"),
    "totals_table":   ("eng+fra", "--psm 6"),
}

_FALLBACK_ROI: Dict[str, Tuple[float, float, float, float]] = {
    # (left, top, right, bottom) in relative percentages
    "invoice_number": (0.11, 0.265, 0.34, 0.33),
    "invoice_date":   (0.36, 0.265, 0.56, 0.33),
    "mode_reglement": (0.11, 0.34, 0.56, 0.385),
    "reference":      (0.11, 0.385, 0.56, 0.43),
    "mf":             (0.56, 0.34, 0.84, 0.385),
    "ccb":            (0.11, 0.43, 0.84, 0.48),
    "lines_table":    (0.09, 0.52, 0.90, 0.63),
    "totals_table":   (0.09, 0.75, 0.90, 0.87),
}


def run_structured_ocr(path: Path) -> tuple[str, Dict[str, str]]:
    # 1) Load, warp, binarize
    pil0    = _load_first_page(path)
    warped  = iops.detect_and_warp(pil0)
    bin_img = iops.binarize_for_text(warped)

    # 2) Full‑page EasyOCR (French + English)
    reader    = easyocr.Reader(['fr', 'en'], gpu=False)
    full_text = "\n".join(reader.readtext(warped, detail=0))

    # 3) Label‑based ROIs for header fields
    label_rects = iops.detect_label_rects(warped)
    roi_texts: Dict[str, str] = {}
    for name, synonyms in iops.LABELS.items():
        if name == "total_ht":
            continue  # handled below
        rect = iops.find_label_box(label_rects, synonyms)
        if rect:
            crop = _crop_to_right_of_label(bin_img, rect)
            lang, cfg = _ROI_TESS_CFG[name]
            roi_texts[name] = _ocr_crop_tess(crop, lang, cfg)

    # 4) Totals region (structured table)
    rect_totals = iops.find_label_box(label_rects, iops.LABELS["total_ht"])
    if rect_totals:
        crop = _crop_totals_region(bin_img, rect_totals)
        lang, cfg = _ROI_TESS_CFG["totals_table"]
        roi_texts["totals_table"] = _ocr_crop_tess(crop, lang, cfg)

    # 5) Fallback %-box crops for any missing ROIs
    tess_ok = bool(shutil.which("tesseract"))
    for name, pct_box in _FALLBACK_ROI.items():
        if name in roi_texts:
            continue
        crop = iops.crop_percent(bin_img, iops.expand_box(pct_box, pad=0.01))
        lang, cfg = _ROI_TESS_CFG.get(name, ("eng+fra", "--psm 6"))
        roi_texts[name] = _ocr_crop_tess(crop, lang, cfg) if tess_ok else ""

    return full_text, roi_texts


# --------------------------------------------------------------------------- #
#                          Utility functions                                  #
# --------------------------------------------------------------------------- #

def _load_first_page(path: Path) -> Image.Image:
    """Open the first page of a PDF or image file as a PIL Image."""
    return Image.open(path)


def _ocr_crop_tess(crop: Image.Image, lang: str, config: str) -> str:
    """Run Tesseract on a cropped region and return the stripped string."""
    import pytesseract
    return pytesseract.image_to_string(crop, lang=lang, config=config).strip()


def _crop_to_right_of_label(
    img: Image.Image,
    rect: Tuple[int, int, int, int]
) -> Image.Image:
    """Crop from the label’s right edge all the way to the image’s right boundary."""
    l, t, r, b = rect
    return img.crop((r, t, img.width, b))


def _crop_totals_region(
    img: Image.Image,
    rect: Tuple[int, int, int, int]
) -> Image.Image:
    """Slightly expand the totals box and crop it."""
    l, t, r, b = rect
    w, h = r - l, b - t
    return img.crop((l, t, r + int(w * 0.2), b + int(h * 0.5)))
