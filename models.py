"""Data models used internally after OCR/parse.

These are lightweight dataclasses, not DB models. All numeric values
use Decimal for precision. `to_dict()` converts Decimals to str so
JSON serialization is safe.
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional


@dataclass
class InvoiceLine:
    reference: str = ""
    designation: str = ""
    quantity: Decimal = Decimal("0")
    unit_price: Decimal = Decimal("0")
    discount: Decimal = Decimal("0")
    amount_ht: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Decimal):
                d[k] = str(v)
        return d


@dataclass
class InvoiceTotals:
    total_ht: Decimal = Decimal("0")
    total_tax: Decimal = Decimal("0")
    total_ttc: Decimal = Decimal("0")
    net_to_pay: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Decimal):
                d[k] = str(v)
        return d


@dataclass
class InvoiceHeader:
    invoice_number: str = ""
    invoice_date: Optional[date] = None
    customer_name: str = ""
    customer_type: str = ""
    customer_address: str = ""
    mode_reglement: str = ""
    reference: str = ""
    mf: str = ""
    rib: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(d.get("invoice_date"), date):
            d["invoice_date"] = d["invoice_date"].isoformat()
        return d


@dataclass
class Invoice:
    header: InvoiceHeader = field(default_factory=InvoiceHeader)
    lines: List[InvoiceLine] = field(default_factory=list)
    totals: Optional[InvoiceTotals] = None
    raw_text: str = ""
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    source_filename: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "lines": [ln.to_dict() for ln in self.lines],
            "totals": self.totals.to_dict() if self.totals else None,
            "raw_text": self.raw_text,
            "parsed_at": self.parsed_at.isoformat() + "Z",
            "source_filename": self.source_filename,
        }
