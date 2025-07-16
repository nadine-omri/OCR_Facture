"""
OCR service – EasyOCR + Tesseract pipeline tailored for mixed
French/Arabic TunTrust invoices.

Main entry point
----------------
run_structured_ocr(path: pathlib.Path) → tuple[str, dict[str, str]]

Returns
• full_text – raw EasyOCR text for the whole page
• roi_texts – keyed result for each region of interest (MF, RIB, totals …)
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import pytesseract

from .config import Config
from .utils import normalize_whitespace
from . import image_ops as iops

# Optional heavyweight/OS-level dependencies – imported lazily
try:
    import easyocr  # noqa: WPS433
except ImportError:  # pragma: no cover
    easyocr = None  # type: ignore

try:
    from pdf2image import convert_from_path  # noqa: WPS433
except ImportError:  # pragma: no cover
    convert_from_path = None

# --------------------------------------------------------------------------- #
#                         CONSTANTS / LOOK-UP TABLES                          #
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
    "invoice_number": (0.11, 0.245, 0.34, 0.29),
    "invoice_date":   (0.34, 0.245, 0.56, 0.29),
    "mode_reglement": (0.11, 0.34, 0.56, 0.385),
    "reference":      (0.11, 0.385, 0.56, 0.43),
    "mf":             (0.56, 0.34, 0.84, 0.385),
    "ccb":            (0.11, 0.43, 0.84, 0.48),
    "lines_table":    (0.09, 0.52, 0.90, 0.63),
    "totals_table":   (0.09, 0.75, 0.90, 0.87),
}

_LABELS = {
    "invoice_number": ["numero", "num", "n°", "nº"],
    "invoice_date":   ["date"],
    "mode_reglement": ["mode de reglement", "mode reglement", "mode de règle"],
    "reference":      ["reference", "référence", "ref"],
    "mf":             ["mf"],
    "ccb":            ["ccb", "c.c.b", "rib"],
    "lines_header":   ["reference designation", "référence désignation", "designation"],
    "total_ht":       ["total ht"],
    "total_taxe":     ["total taxe"],
    "total_ttc":      ["total ttc"],
    "net":            ["net a payer", "net payer"],
}

# Cached EasyOCR readers
_READER_LATIN = None
_READER_ARABIC = None

# --------------------------------------------------------------------------- #
#                               CORE HELPERS                                  #
# --------------------------------------------------------------------------- #


def _detect_tesseract_cmd() -> str | None:
    # user override
    if Config.TESSERACT_CMD:
        return Config.TESSERACT_CMD
    # typical Linux/Mac locations
    for cmd in ("tesseract", "/usr/local/bin/tesseract"):
        if shutil.which(cmd):
            return cmd
    brew = Path("/opt/homebrew/bin/tesseract")  # Apple Silicon default
    return str(brew) if brew.exists() else None


def _ensure_tesseract_cmd() -> bool:
    """Wire `pytesseract` to the native binary; return False if absent."""
    cmd = _detect_tesseract_cmd()
    if not cmd:
        return False
    pytesseract.pytesseract.tesseract_cmd = cmd
    return True


# --------------------------------------------------------------------------- #
#                              OCR LOW-LEVEL                                  #
# --------------------------------------------------------------------------- #


def _ocr_crop_tess(img: Image.Image, langs: str, cfg: str = "") -> str:
    """Run Tesseract on *img* with optional up-scaling for better accuracy.

    * If the crop width is < 800 px we upscale ×2 using bicubic sampling. This
      appreciably improves digit recognition on mobile photos without adding
      much latency.
    * `cfg` falls back to `--psm 6` when not provided.
    """
    w, h = img.size
    if w < 800:
        img = img.resize((w * 2, h * 2), Image.BICUBIC)
    return normalize_whitespace(
        pytesseract.image_to_string(img, lang=langs, config=cfg or "--psm 6")
    )


def _crop_to_right_of_label(
    img: Image.Image,
    label_rect: Tuple[int, int, int, int],
    *,
    frac_x: float = 0.35,
    pad_pct: float = 0.01,
) -> Image.Image:
    """
    Return a crop that starts just to the right of *label_rect*.

    *frac_x* – width of the crop as percentage of full-image width.
    """
    l, t, r, b = label_rect
    w_img, _ = img.size
    x0 = int(r + pad_pct * w_img)
    x1 = int(min(w_img, x0 + frac_x * w_img))
    return img.crop((x0, t, x1, b + int(pad_pct * w_img)))


def _bbox_rect(bbox) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _match_label(txt_norm: str, patterns: Iterable[str]) -> bool:
    """Return *True* when *txt_norm* is close to any candidate pattern.

    Fast path → substring check.
    Slow path → fuzzy ratio ≥ 0.8 (SequenceMatcher) to absorb OCR glitches.
    """
    for pat in patterns:
        if pat in txt_norm:
            return True
        if SequenceMatcher(None, txt_norm, pat).ratio() >= 0.8:
            return True
    return False


def _norm_txt(s: str) -> str:
    s = s.lower()
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    return s


def _get_easy_readers() -> List["easyocr.Reader"]:
    """Lazily instantiate and cache the two EasyOCR readers."""
    global _READER_LATIN, _READER_ARABIC
    if easyocr is None:
        return []
    if _READER_LATIN is None:
        _READER_LATIN = easyocr.Reader(["fr", "en"], gpu=False)
    if _READER_ARABIC is None:
        _READER_ARABIC = easyocr.Reader(["ar", "en"], gpu=False)
    return [_READER_LATIN, _READER_ARABIC]


# --------------------------------------------------------------------------- #
#                             PAGE-LEVEL LOADERS                              #
# --------------------------------------------------------------------------- #


def _pdf_to_images(path: Path) -> List[Image.Image]:
    if convert_from_path is None:
        raise RuntimeError("pdf2image not installed – cannot OCR PDFs.")
    return convert_from_path(str(path))


def _load_first_page(path: Path) -> Image.Image:
    """
    Open *path* and return a PIL.Image.

    * PDF → convert only the first page
    * Image → open as-is (RGB)
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        page = _pdf_to_images(path)[0]
        return page.convert("RGB")
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------- #
#           ROI CROPPING UTILS (between labels / fixed-box fallbacks)         #
# --------------------------------------------------------------------------- #


def _crop_table_between(
    img: Image.Image, top_rect: Tuple[int, int, int, int], bottom_rect: Tuple[int, int, int, int]
) -> Image.Image:
    l, _, r, _ = top_rect
    _, t2, _, _ = bottom_rect
    w_img, h_img = img.size
    return img.crop((max(0, l - 10), top_rect[1], min(w_img, r + 10), min(h_img, t2 + 120)))


def _crop_totals_region(
    img: Image.Image, total_ht_rect: Tuple[int, int, int, int], *, pad_pct: float = 0.01
) -> Image.Image:
    l, t, r, b = total_ht_rect
    w_img, h_img = img.size
    y0 = max(0, b - int(0.05 * h_img))
    y1 = min(h_img, b + int(0.10 * h_img))
    return img.crop((max(0, l - int(0.02 * w_img)), y0, min(w_img, r + int(0.5 * w_img)), y1))


# --------------------------------------------------------------------------- #
#                           PUBLIC SERVICE ENTRY POINT                        #
# --------------------------------------------------------------------------- #


def run_structured_ocr(path: Path) -> tuple[str, Dict[str, str]]:
    """High-level method – returns full EasyOCR text and ROI dict."""
    tess_ok = _ensure_tesseract_cmd()

    # 1. Load + pre-process
    pil0 = _load_first_page(path)
    warped = iops.detect_and_warp(pil0)
    bin_img = iops.binarize_for_text(warped)  # grayscale (PIL “L”)

    # 2. EasyOCR on full page – merge Arabic + Latin detections
    easy_results = []
    for reader in _get_easy_readers():
        easy_results.extend(
            reader.readtext(np.array(bin_img), detail=1, paragraph=False)
        )
    full_text = "\n".join(r[1] for r in easy_results)

    # 3. Detect label positions
    label_rects: Dict[str, Tuple[int, int, int, int]] = {}
    for bbox, text, _ in easy_results:
        n = _norm_txt(text)
        rect = _bbox_rect(bbox)
        for key, patterns in _LABELS.items():
            if key not in label_rects and _match_label(n, patterns):
                label_rects[key] = rect

    # 4. ROI-specific crops + re-OCR
    roi_texts: Dict[str, str] = {}
    if tess_ok:
        # Simple right-of-label fields
        def _grab(name: str, frac: float):
            if name in label_rects:
                crop = _crop_to_right_of_label(bin_img, label_rects[name], frac_x=frac)
                roi_texts[name] = _ocr_crop_tess(crop, *_ROI_TESS_CFG[name])

        _grab("invoice_number", 0.30)
        _grab("invoice_date",   0.18)
        _grab("mode_reglement", 0.40)
        _grab("reference",      0.40)
        _grab("mf",             0.20)
        _grab("ccb",            0.60)

        # Lines table
        if "lines_header" in label_rects and "total_ht" in label_rects:
            crop = _crop_table_between(
                bin_img, label_rects["lines_header"], label_rects["total_ht"]
            )
            roi_texts["lines_table"] = _ocr_crop_tess(crop, *_ROI_TESS_CFG["lines_table"])

        # Totals row
        if "total_ht" in label_rects:
            crop = _crop_totals_region(bin_img, label_rects["total_ht"])
            roi_texts["totals_table"] = _ocr_crop_tess(
                crop, *_ROI_TESS_CFG["totals_table"]
            )

    # 5. Fallback %-box crops for anything still missing
    for name, pct_box in _FALLBACK_ROI.items():
        if name in roi_texts:
            continue
        crop = iops.crop_percent(bin_img, iops.expand_box(pct_box, pad=0.01))
        lang, cfg = _ROI_TESS_CFG.get(name, ("eng+fra", "--psm 6"))
        roi_texts[name] = _ocr_crop_tess(crop, lang, cfg) if tess_ok else ""

    return full_text, roi_texts
