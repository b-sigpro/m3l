# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import tempfile
import time
import uuid


class TimedTemporaryFiles:
    """Create temporary output paths and lazily remove expired files."""

    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self._temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp_dir.name)

    def create_path(self, suffix: str = "") -> Path:
        self._cleanup_expired()
        return self.directory / f"{uuid.uuid4().hex}{suffix}"

    def _cleanup_expired(self) -> None:
        cutoff = time.time() - self.ttl

        for path in self.directory.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
