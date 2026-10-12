"""Atomic file writes for files the API serves while other requests may be reading them.

A plain write_text truncates the file first, so a concurrent reader (the UI polls the
motion library every few seconds) can see an empty or half-written file. Writing a
temporary file in the same directory and os.replace()-ing it in is atomic on POSIX and
Windows: readers see either the old file or the new one, never a mix.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    # Unique temp name, so two concurrent writers never share (and corrupt) one temp file.
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))
