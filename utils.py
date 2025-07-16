"""Parsing helpers: numbers, dates, whitespace cleanup."""

import re
from datetime import date
from decimal import Decimal, InvalidOperation


_FR_DATE_PAT = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
_NUM_CLEAN = re.compile(r"[^\d,.\-]")


def parse_fr_number(text: str) -> Decimal:
    """Parse French style numbers into Decimal.

    Examples
    --------
    '325,000' -> 325.000
    '1 234,56' -> 1234.56
    Fallback handles slightly dirty OCR strings.
    """
    if not text:
        return Decimal("0")
    t = text.strip().replace(" ", "")
    t = "".join(ch for ch in t if ch.isdigit() or ch in ",.-")
    if t.count(",") == 1 and t.rfind(",") > t.rfind("."):
        t = t.replace(".", "")
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return Decimal(t)
    except InvalidOperation:
        return Decimal("0")


def parse_fr_date(text: str) -> date | None:
    """Parse dd/mm/yy(yy) or dd-mm-yy(yy)."""
    if not text:
        return None
    m = _FR_DATE_PAT.search(text.strip())
    if not m:
        return None
    dd, mm, yy = m.groups()
    day = int(dd)
    month = int(mm)
    year = int(yy)
    if year < 100:
        year += 2000 if year < 70 else 1900
    try:
        return date(year, month, day)
    except ValueError:
        return None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_line_containing(lines: list[str], *keywords: str) -> str:
    """Return the first line that contains all keywords (case-insensitive)."""
    lowered = [ln.lower() for ln in lines]
    kws = [kw.lower() for kw in keywords]
    for idx, ln in enumerate(lowered):
        if all(kw in ln for kw in kws):
            return lines[idx]
    return ""


def extract_after_colon(text: str) -> str:
    """Return substring after the first colon if present."""
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text.strip()
