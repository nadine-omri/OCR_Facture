"""Flask application factory and routes.

Upgraded to structured OCR: we call `run_structured_ocr()` which returns full
text plus ROI texts. We pass both into the parser.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Dict

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    abort,
    flash,
    jsonify,
)
from werkzeug.utils import secure_filename

from .config import get_config, Config
from .ocr_service import run_structured_ocr
from .parser_tuntrust import parse_invoice
from .models import Invoice

# in-memory store of parsed invoices keyed by uuid str
_INVOICE_STORE: Dict[str, Invoice] = {}

# absolute project root (directory that contains this file)
ROOT = Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"
TEMPLATE_DIR = ROOT / "templates"


def allowed_file(filename: str) -> bool:
    """Check extension allowlist."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def create_app(config_name: str | None = None) -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
        template_folder=str(TEMPLATE_DIR),
    )
    cfg_cls = get_config(config_name)
    app.config.from_object(cfg_cls)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("upload.html")

    @app.route("/upload", methods=["POST"])
    def upload():
        if "file" not in request.files:
            flash("No file part")
            return redirect(url_for("index"))
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("File type not allowed")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        dest = Path(app.config["UPLOAD_FOLDER"]) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        file.save(dest)

        try:
            full_text, roi_texts = run_structured_ocr(dest)
            if not full_text and not roi_texts:
                flash("Tesseract not found so OCR skipped. Showing empty parse.")
            invoice = parse_invoice(full_text, source_filename=filename, roi_texts=roi_texts)
        except Exception as exc:
            return render_template("error.html", message=str(exc)), 500

        inv_id = str(uuid.uuid4())
        _INVOICE_STORE[inv_id] = invoice
        return redirect(url_for("view_invoice", invoice_id=inv_id))

    @app.route("/invoice/<invoice_id>", methods=["GET"])
    def view_invoice(invoice_id: str):
        invoice = _INVOICE_STORE.get(invoice_id)
        if not invoice:
            abort(404)
        return render_template("invoice.html", invoice=invoice, invoice_id=invoice_id)

    @app.route("/invoice/<invoice_id>/json", methods=["GET"])
    def invoice_json(invoice_id: str):
        invoice = _INVOICE_STORE.get(invoice_id)
        if not invoice:
            abort(404)
        return jsonify(invoice.to_dict())

    @app.route("/invoice/<invoice_id>/download-json", methods=["GET"])
    def invoice_download_json(invoice_id: str):
        invoice = _INVOICE_STORE.get(invoice_id)
        if not invoice:
            abort(404)
        data = json.dumps(invoice.to_dict(), indent=2).encode("utf-8")
        return send_file(
            io.BytesIO(data),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"invoice_{invoice_id}.json",
        )

    return app
