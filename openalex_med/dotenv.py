"""Tiny dependency-free ``.env`` loader.

We don't pull in ``python-dotenv`` because the format we need is trivial and
adding a hard dep just to parse ``KEY=VALUE`` lines is overkill.

Behaviour
---------
* Lines starting with ``#`` and blank lines are ignored.
* Surrounding single or double quotes around the value are stripped.
* By default we use :func:`os.environ.setdefault` so values already in the
  process environment win — pass ``override=True`` to flip that.
* A missing ``.env`` file is *not* an error; the loader is a no-op.

The loader walks upwards from the current working directory looking for a
``.env`` so that running ``python -m openalex_med.embed`` from anywhere in
the project tree picks up the same file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def find_dotenv(start: str | Path | None = None, *, name: str = ".env") -> Path | None:
    """Walk up from ``start`` looking for ``name``; return its path or None."""
    here = Path(start or Path.cwd()).resolve()
    for cand in (here, *here.parents):
        p = cand / name
        if p.is_file():
            return p
    return None


def load_dotenv(
    path: str | Path | None = None,
    *,
    override: bool = False,
    quiet: bool = True,
) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into ``os.environ``.

    Returns the dict of pairs that were applied (useful for tests).
    """
    if path is None:
        located = find_dotenv()
        if located is None:
            if not quiet:
                log.info("No .env file found")
            return {}
        path = located

    p = Path(path)
    if not p.is_file():
        if not quiet:
            log.info("No .env file at %s", p)
        return {}

    applied: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip matching surrounding quotes.
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ("'", '"')
        ):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    if not quiet:
        log.info("Loaded %d entries from %s", len(applied), p)
    return applied
