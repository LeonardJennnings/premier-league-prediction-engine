"""Configure project-wide logging."""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    """Configure a simple root logger once.

    Level comes from the argument, else LOG_LEVEL env var, else INFO.
    """
    name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    numeric = getattr(logging, name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(numeric)
        return

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
