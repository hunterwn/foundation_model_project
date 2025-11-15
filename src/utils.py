from __future__ import annotations

import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def slugify(value: str, max_length: int = 80) -> str:
    """Return a filesystem-safe slug."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if len(value) > max_length:
        value = value[:max_length].rstrip("-")
    return value or "prompt"


def ensure_dir(path: os.PathLike | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunked(iterable: Iterable, size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
