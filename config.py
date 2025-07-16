"""Application configuration objects.

All paths are derived relative to this file so things work whether
run from PyCharm, flask CLI, or gunicorn.
"""

import os
from pathlib import Path
from typing import Set


class Config:
    """Base configuration shared by all environments."""

    BASE_DIR = Path(__file__).resolve().parent
    # project root is the directory that contains this config.py
    PROJECT_ROOT = BASE_DIR

    # uploads live under PROJECT_ROOT/uploads
    UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Max upload size (default 8 MB)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 8 * 1024 * 1024))

    # Allowed file extensions
    ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "pdf", "tif", "tiff"}

    # Tesseract languages: TunTrust invoices have French + Arabic labels
    TESS_LANG = os.getenv("TESS_LANG", "fra+ara")

    # Optional explicit path to tesseract binary
    TESSERACT_CMD = os.getenv("TESSERACT_CMD")

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-me")


class DevConfig(Config):
    DEBUG = True
    TESTING = False


class TestConfig(Config):
    DEBUG = False
    TESTING = True
    # use temp upload folder in tests
    UPLOAD_FOLDER = Config.PROJECT_ROOT / "test_uploads"
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


class ProdConfig(Config):
    DEBUG = False
    TESTING = False
    # override via env in production


config_by_name = {
    "development": DevConfig,
    "testing": TestConfig,
    "production": ProdConfig,
}


def get_config(name: str | None = None):
    """Return a config class by name; defaults to FLASK_ENV or DevConfig."""
    if not name:
        name = os.getenv("FLASK_ENV", "development")
    return config_by_name.get(name, DevConfig)
