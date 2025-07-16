"""TunTrust / STB invoice parser with ROI support.

We try structured extraction from ROI OCR results first (highest quality),
then fill any missing fields by scanning the full OCR text with regex
fallbacks (lenient, noisy).

If line / totals parsing from ROIs fails, fallback regex runs over full text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .models import Invoice, InvoiceHeader, InvoiceLine, InvoiceTotals
from .utils import (
    parse_fr_date,
    parse_fr_number,
    find_line_containing,
    normalize_whitespace,
)

# header regex fallbacks
_RE_NUM = re.compile(r"num[ée]ro\s*[:\-]?\s*([A-Z0-9\-_/]+)", re.IGNORECASE)
_RE_DATE = re.compile(r"date\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})", re.IGNORECASE)
_RE_MODE = re.compile(r"mode\s+de\s+r[ée]glement\s*[:\-]?\s*(.+)", re.IGNORECASE)
_RE_REF = re.compile(r"r[ée]f[ée]rence\s*[:\-]?\s*([A-Z0-9\-_/ ]+)", re.IGNORECASE)
_RE_MF = re.compile(r"\bMF\s*[:\-]?\s*([A-Z0-9]+)", re.IGNORECASE)
_RE_CCB = re.compile(r"(?:C\.?C\.?B\.?|RIB)\s*[:\-]?\s*([0-9\s]+)", re.IGNORECASE)

# totals regex fallbacks
_RE_TOTAL_HT = re.compile(r"Total\s+HT\s*([0-9.,\s]+)", re.IGNORECASE)
_RE_TOTAL_TAX = re.compile(r"Total\s+Taxe\s*([0-9.,\s]+)", re.IGNORECASE)
_RE_TOTAL_TTC = re.compile(r"Total\s+TTC\s*([0-9.,\s]+)", re.IGNORECASE)
_RE_NET = re.compile(r"(?:NET\s+A\s+PAYER|NET\s+PAYABLE)\s*([0-9.,\s]+)", re.IGNORECASE)

# line row heuristic pattern (ref desig qty pu remise mht)
_LINE_PAT = re.compile(
    r"""
    ^\s*
    (?P<ref>\S+)\s+                                   # reference token
    (?P<desig>.+?)\s+                                 # designation greedy
    (?P<qty>[0-9][0-9.,]*)\s+                         # quantity
    (?P<pu>[0-9][0-9.,]*)\s+                          # unit price
    (?P<rem>[0-9][0-9.,]*)\s+                         # remise
    (?P<mht>[0-9][0-9.,]*)\s*$                        # montant ht
    """,
    re.VERBOSE,
)


# ------------------------------------------------------------------ ROI helpers
def _clean_roi_text(txt: str) -> str:
    if not txt:
        return ""
    return normalize_whitespace(txt)


def _extract_invoice_number(txt: str) -> str:
    # just take first contiguous token of A-Z0-9
    m = re.search(r"[A-Z0-9][A-Z0-9/_-]*", txt.upper())
    return m.group(0) if m else ""


def _extract_mf(txt: str) -> str:
    m = re.search(r"[A-Z0-9]+", txt.upper())
    return m.group(0) if m else ""


def _extract_ccb(txt: str) -> str:
    m = re.findall(r"\d+", txt)
    return "".join(m)


def _parse_lines_roi(txt: str) -> List[InvoiceLine]:
    rows = []
    for raw in txt.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # skip column headers if present
        low = ln.lower()
        if "référence" in low and "désignation" in low:
            continue
        m = _LINE_PAT.match(ln)
        if not m:
            continue
        gd = m.groupdict()
        rows.append(
            InvoiceLine(
                reference=gd["ref"].strip(),
                designation=gd["desig"].strip(),
                quantity=parse_fr_number(gd["qty"]),
                unit_price=parse_fr_number(gd["pu"]),
                discount=parse_fr_number(gd["rem"]),
                amount_ht=parse_fr_number(gd["mht"]),
            )
        )
    return rows


def _parse_totals_roi(txt: str) -> Optional[InvoiceTotals]:
    if not txt.strip():
        return None
    # try explicit labels
    m_ht = _RE_TOTAL_HT.search(txt)
    m_tax = _RE_TOTAL_TAX.search(txt)
    m_ttc = _RE_TOTAL_TTC.search(txt)
    m_net = _RE_NET.search(txt)

    # if none matched but there are numeric rows, pick last 1-4 numbers
    if not any([m_ht, m_tax, m_ttc, m_net]):
        nums = re.findall(r"[0-9][0-9.,]*", txt)
        if len(nums) >= 4:
            ht, tax, ttc, net = nums[-4:]
            return InvoiceTotals(
                total_ht=parse_fr_number(ht),
                total_tax=parse_fr_number(tax),
                total_ttc=parse_fr_number(ttc),
                net_to_pay=parse_fr_number(net),
            )
        return None

    tot = InvoiceTotals()
    if m_ht:
        tot.total_ht = parse_fr_number(m_ht.group(1))
    if m_tax:
        tot.total_tax = parse_fr_number(m_tax.group(1))
    if m_ttc:
        tot.total_ttc = parse_fr_number(m_ttc.group(1))
    if m_net:
        tot.net_to_pay = parse_fr_number(m_net.group(1))
    return tot


# -------------------------------------------------------------- fallback parse
def _parse_header_fallback(text_lines: List[str], hdr: InvoiceHeader) -> InvoiceHeader:
    if not hdr.invoice_number:
        for ln in text_lines:
            m = _RE_NUM.search(ln)
            if m:
                hdr.invoice_number = m.group(1).strip()
                break

    if hdr.invoice_date is None:
        for ln in text_lines:
            m = _RE_DATE.search(ln)
            if m:
                hdr.invoice_date = parse_fr_date(m.group(1))
                break

    if not hdr.mode_reglement:
        for ln in text_lines:
            m = _RE_MODE.search(ln)
            if m:
                hdr.mode_reglement = normalize_whitespace(m.group(1))
                break

    if not hdr.reference:
        for ln in text_lines:
            m = _RE_REF.search(ln)
            if m:
                hdr.reference = m.group(1).strip()
                break

    if not hdr.mf:
        for ln in text_lines:
            m = _RE_MF.search(ln)
            if m:
                hdr.mf = m.group(1).strip()
                break

    if not hdr.rib:
        for ln in text_lines:
            m = _RE_CCB.search(ln)
            if m:
                hdr.rib = m.group(1).replace(" ", "").strip()
                break

    # name / type / address heuristics
    if not hdr.customer_name:
        cust_line = find_line_containing(text_lines, "tunisi", "banque")
        if not cust_line:
            cust_line = find_line_containing(text_lines, "banque")
        hdr.customer_name = cust_line.strip()
        # type/address from following lines
        try:
            idx = text_lines.index(cust_line)
        except ValueError:
            idx = -1
        if idx >= 0 and idx + 1 < len(text_lines):
            hdr.customer_type = text_lines[idx + 1].strip()
        if idx >= 0 and idx + 2 < len(text_lines):
            cand = text_lines[idx + 2]
            if any(k in cand.lower() for k in ["rue", "avenue", "tun", "bp"]):
                hdr.customer_address = cand.strip()

    return hdr


def _parse_lines_fallback(text_lines: List[str]) -> List[InvoiceLine]:
    items: List[InvoiceLine] = []
    start_idx = -1
    for i, ln in enumerate(text_lines):
        low = ln.lower()
        if "référence" in low and "désignation" in low:
            start_idx = i + 1
            break
    if start_idx < 0:
        return items

    for ln in text_lines[start_idx:]:
        if not ln.strip():
            break
        low = ln.lower()
        if "total ht" in low or "total taxe" in low or "net a payer" in low:
            break
        m = _LINE_PAT.match(ln)
        if not m:
            continue
        gd = m.groupdict()
        items.append(
            InvoiceLine(
                reference=gd["ref"].strip(),
                designation=gd["desig"].strip(),
                quantity=parse_fr_number(gd["qty"]),
                unit_price=parse_fr_number(gd["pu"]),
                discount=parse_fr_number(gd["rem"]),
                amount_ht=parse_fr_number(gd["mht"]),
            )
        )
    return items


def _parse_totals_fallback(text: str) -> Optional[InvoiceTotals]:
    m_ht = _RE_TOTAL_HT.search(text)
    m_tax = _RE_TOTAL_TAX.search(text)
    m_ttc = _RE_TOTAL_TTC.search(text)
    m_net = _RE_NET.search(text)

    if not any([m_ht, m_tax, m_ttc, m_net]):
        return None

    tot = InvoiceTotals()
    if m_ht:
        tot.total_ht = parse_fr_number(m_ht.group(1))
    if m_tax:
        tot.total_tax = parse_fr_number(m_tax.group(1))
    if m_ttc:
        tot.total_ttc = parse_fr_number(m_ttc.group(1))
    if m_net:
        tot.net_to_pay = parse_fr_number(m_net.group(1))
    return tot


# -------------------------------------------------------------------- main API
def parse_invoice(
    text: str,
    source_filename: str = "",
    roi_texts: Dict[str, str] | None = None,
) -> Invoice:
    """Produce an Invoice datamodel from OCR text and ROI texts."""
    roi_texts = roi_texts or {}
    # clean ROI
    roi_clean = {k: _clean_roi_text(v) for k, v in roi_texts.items()}

    hdr = InvoiceHeader()
    # structured header from ROI
    if roi_clean.get("invoice_number"):
        hdr.invoice_number = _extract_invoice_number(roi_clean["invoice_number"])
    if roi_clean.get("invoice_date"):
        hdr.invoice_date = parse_fr_date(roi_clean["invoice_date"])
    if roi_clean.get("mode_reglement"):
        hdr.mode_reglement = roi_clean["mode_reglement"]
    if roi_clean.get("reference"):
        hdr.reference = roi_clean["reference"]
    if roi_clean.get("mf"):
        hdr.mf = _extract_mf(roi_clean["mf"])
    if roi_clean.get("ccb"):
        hdr.rib = _extract_ccb(roi_clean["ccb"])

    # fallback parse lines -> list of strings
    raw_lines = [normalize_whitespace(ln) for ln in text.splitlines()]
    raw_lines = [ln for ln in raw_lines if ln]

    hdr = _parse_header_fallback(raw_lines, hdr)

    # line items
    items = _parse_lines_roi(roi_clean.get("lines_table", "")) if roi_clean.get("lines_table") else []
    if not items:
        items = _parse_lines_fallback(raw_lines)

    # totals
    totals = _parse_totals_roi(roi_clean.get("totals_table", "")) if roi_clean.get("totals_table") else None
    if not totals:
        totals = _parse_totals_fallback(text)

    return Invoice(
        header=hdr,
        lines=items,
        totals=totals,
        raw_text=text,
        source_filename=source_filename,
    )
