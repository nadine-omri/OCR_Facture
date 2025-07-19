# parser_tuntrust.py
"""
TunTrust / STB invoice parser with ROI support.

1) Try structured extraction from ROI OCR results
2) Fill missing header fields via regex or line‑following fallback
3) Parse line‐items & totals (ROI first, then fallback)
"""

from __future__ import annotations
import re
import unicodedata
from typing import Dict, List
from decimal import Decimal

from .models import Invoice, InvoiceHeader, InvoiceLine, InvoiceTotals
from .utils import find_line_containing, normalize_whitespace, parse_fr_date

# --------------------------------------------------------------------------- #
#                          HEADER REGEX FALLBACKS                             #
# --------------------------------------------------------------------------- #
_RE_NUM        = re.compile(r"num[ée]ro\s*[:\-]?\s*([A-Z0-9\-_/]+)", re.IGNORECASE)
_RE_DATE       = re.compile(r"date\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})", re.IGNORECASE)
_RE_MODE       = re.compile(r"mode\s+de\s+r[ée]glement\s*[:\-]?\s*(.+)", re.IGNORECASE)
_RE_REF        = re.compile(r"r[ée]f[ée]rence\s*[:\-]?\s*([A-Z0-9\-_/ ]+)", re.IGNORECASE)
_RE_MF         = re.compile(r"\bMF\s*[:\-]?\s*([A-Z0-9]+)", re.IGNORECASE)
_RE_CCB        = re.compile(r"(?:C\.?C\.?B\.?|RIB)\s*[:\-]?\s*([0-9\s]+)", re.IGNORECASE)

# --------------------------------------------------------------------------- #
#                         TOTALS REGEX FALLBACKS                              #
# --------------------------------------------------------------------------- #
_RE_TOTAL_HT   = re.compile(r"Total\W*Ht\W*([\d.,\s]+)", re.IGNORECASE)
_RE_TOTAL_TAX  = re.compile(r"Total\W*Taxe\W*([\d.,\s]+)", re.IGNORECASE)
_RE_TOTAL_TTC  = re.compile(r"Total\W*TTC\W*([\d.,\s]+)", re.IGNORECASE)


def _remove_accents(text: str) -> str:
    """Strip diacritics for easier matching."""
    return ''.join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _parse_header_fallback(text_lines: List[str], hdr: InvoiceHeader) -> InvoiceHeader:
    """Fill any missing header fields by scanning full‐text lines."""
    # invoice_number
    if not hdr.invoice_number:
        prev = ""
        for ln in text_lines:
            m = _RE_NUM.search(ln)
            if m:
                hdr.invoice_number = m.group(1).strip()
                break
            if prev.lower().startswith("num"):
                cand = ln.strip()
                if re.fullmatch(r"[A-Z0-9\-_\/]+", cand):
                    hdr.invoice_number = cand
                    break
            prev = ln

    # invoice_date
    if hdr.invoice_date is None:
        prev = ""
        for ln in text_lines:
            m = _RE_DATE.search(ln)
            if m:
                hdr.invoice_date = parse_fr_date(m.group(1))
                break
            if prev.lower().strip() == "date":
                digits = re.sub(r"\D", "", ln)
                if len(digits) in (6, 8):
                    fmt = (
                        f"{digits[0:2]}/{digits[2:4]}/{digits[4:6]}"
                        if len(digits) == 6
                        else f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
                    )
                    hdr.invoice_date = parse_fr_date(fmt)
                    break
            prev = ln

    # mode_reglement
    if not hdr.mode_reglement:
        for ln in text_lines:
            m = _RE_MODE.search(ln)
            if m:
                hdr.mode_reglement = normalize_whitespace(m.group(1))
                break

    # reference
    if not hdr.reference:
        for ln in text_lines:
            m = _RE_REF.search(ln)
            if m:
                hdr.reference = m.group(1).strip()
                break

    # MF
    if not hdr.mf:
        for ln in text_lines:
            m = _RE_MF.search(ln)
            if m:
                hdr.mf = m.group(1).strip()
                break

    # RIB / CCB
    if not hdr.rib:
        for ln in text_lines:
            m = _RE_CCB.search(ln)
            if m:
                hdr.rib = m.group(1).replace(" ", "").strip()
                break

    # customer_name & address
    if not hdr.customer_name:
        cust = find_line_containing(text_lines, "tunisi", "banque") \
               or find_line_containing(text_lines, "banque")
        if cust:
            hdr.customer_name = cust.strip()
            idx = text_lines.index(cust)
            addr = []
            for nxt in text_lines[idx + 1 :]:
                if nxt.lower().startswith(("tél", "fax", "mf", "mode", "réfé")):
                    break
                addr.append(nxt)
            hdr.customer_address = " ".join(addr).strip()

    return hdr


# --------------------------------------------------------------------------- #
#                         ROI & FALLBACK PARSERS                              #
# --------------------------------------------------------------------------- #

def _parse_lines_roi(text: str) -> List[InvoiceLine]:
    """Stub for ROI‐based line‐item parsing – replace with your real logic."""
    # TODO: implement actual table parsing from `text`
    return []


def _parse_lines_fallback(lines: List[str]) -> List[InvoiceLine]:
    """Stub for fallback line‐item parsing – replace with your real logic."""
    # TODO: scan `lines` for items, split columns, construct InvoiceLine(...)
    return []


def _parse_totals_roi(text: str) -> InvoiceTotals | None:
    """Stub for ROI‐based totals parsing – replace with your real logic."""
    # TODO: parse total_ht, total_tax, total_ttc, net_to_pay from `text`
    return None


def _parse_totals_fallback(text: str) -> InvoiceTotals:
    """Fallback totals: use regex to pull HT, Taxe, TTC; net = TTC."""
    def _clean_val(m: re.Match) -> Decimal:
        s = re.sub(r"[^\d,\.]", "", m.group(1)).replace(" ", "").replace(",", "")
        return Decimal(s) if s else Decimal(0)

    m_ht  = _RE_TOTAL_HT.search(text)
    m_tax = _RE_TOTAL_TAX.search(text)
    m_ttc = _RE_TOTAL_TTC.search(text)

    total_ht  = _clean_val(m_ht) if m_ht else Decimal(0)
    total_tax = _clean_val(m_tax) if m_tax else Decimal(0)
    total_ttc = _clean_val(m_ttc) if m_ttc else Decimal(0)

    return InvoiceTotals(
        total_ht=total_ht,
        total_tax=total_tax,
        total_ttc=total_ttc,
        net_to_pay=total_ttc,
    )


def parse_invoice(
    text: str,
    source_filename: str = "",
    roi_texts: Dict[str, str] | None = None,
) -> Invoice:
    """
    Main entry point for parsing a TunTrust invoice.

    Parameters
    ----------
    text : str
      Full raw OCR text (line breaks preserved).
    source_filename : str
      Name/path of the source image/PDF.
    roi_texts : dict
      The ROI‐extracted strings from ocr_service.run_structured_ocr().

    Returns
    -------
    Invoice
    """
    # 1) Clean ROI results
    roi_clean = {k: _remove_accents(v).strip() for k, v in (roi_texts or {}).items()}
    hdr = InvoiceHeader()

    # 2) Populate from ROI if available
    if roi_clean.get("invoice_number"):
        hdr.invoice_number = roi_clean["invoice_number"]
    if roi_clean.get("invoice_date"):
        hdr.invoice_date = parse_fr_date(roi_clean["invoice_date"])
    if roi_clean.get("mode_reglement"):
        hdr.mode_reglement = roi_clean["mode_reglement"]
    if roi_clean.get("reference"):
        hdr.reference = roi_clean["reference"]
    if roi_clean.get("mf"):
        hdr.mf = roi_clean["mf"]
    if roi_clean.get("ccb"):
        hdr.rib = roi_clean["ccb"]

    # 3) Fallback header parsing on full text
    raw_lines = [normalize_whitespace(ln) for ln in text.splitlines() if ln.strip()]
    hdr = _parse_header_fallback(raw_lines, hdr)

    # 4) Parse line‐items & totals
    lines  = _parse_lines_roi(roi_clean.get("lines_table", "")) \
             or _parse_lines_fallback(raw_lines)
    totals = _parse_totals_roi(roi_clean.get("totals_table", "")) \
             or _parse_totals_fallback(text)

    # 5) Build and return
    return Invoice(
        header=hdr,
        lines=lines,
        totals=totals,
        raw_text=text,
        source_filename=source_filename,
    )
