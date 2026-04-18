"""Reconstruct OpenAlex abstracts from their inverted index.

OpenAlex stores abstracts as an `abstract_inverted_index`:

    {
        "Despite": [0],
        "growing":  [1],
        "evidence": [2, 10],
        ...
    }

Each key is a token; the value is the list of positions at which the token
appears in the original abstract. To rebuild the abstract we invert the
mapping: place every token at each of its positions in a sparse list, then
join the slots with single spaces.

Notes
-----
* OpenAlex strips abstracts for some works (legal / publisher reasons). When
  the field is `None` or `{}` we return `None` so callers can decide how to
  handle missing abstracts.
* OpenAlex tokenisation is whitespace based. The original whitespace and
  punctuation spacing is *not* preserved exactly, but the resulting string is
  what every downstream OpenAlex consumer uses and what embedding models are
  generally trained against.
* Positions are 0-indexed but occasionally contain gaps (e.g. tokens that the
  indexer skipped). We fill gaps with the empty string and then collapse
  multiple spaces, so the output is always a clean, single-spaced string.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

_MULTI_WS = re.compile(r"\s+")


def reconstruct_abstract(
    inverted_index: Mapping[str, Sequence[int]] | None,
) -> str | None:
    """Reconstruct a flat abstract string from an OpenAlex inverted index.

    Parameters
    ----------
    inverted_index:
        The `abstract_inverted_index` field from an OpenAlex work, or `None`.

    Returns
    -------
    The reconstructed abstract as a single string, or `None` if the index is
    missing/empty.
    """
    if not inverted_index:
        return None

    max_pos = -1
    total = 0
    for positions in inverted_index.values():
        for p in positions:
            if p > max_pos:
                max_pos = p
            total += 1

    if max_pos < 0 or total == 0:
        return None

    slots: list[str] = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        for p in positions:
            if 0 <= p <= max_pos:
                slots[p] = token

    text = " ".join(s for s in slots if s)
    text = _MULTI_WS.sub(" ", text).strip()
    return text or None


def build_text(
    title: str | None,
    abstract: str | None,
    mode: str = "title_abstract",
) -> str | None:
    """Combine title and abstract according to the requested embedding mode.

    Parameters
    ----------
    title:
        OpenAlex `title` (or `display_name`) field.
    abstract:
        Reconstructed abstract (see :func:`reconstruct_abstract`).
    mode:
        One of ``"title"``, ``"abstract"`` or ``"title_abstract"``.

    Returns
    -------
    The text to embed, or `None` if the chosen fields are unavailable.
    """
    title = (title or "").strip() or None
    abstract = (abstract or "").strip() or None

    if mode == "title":
        return title
    if mode == "abstract":
        return abstract
    if mode == "title_abstract":
        if title and abstract:
            return f"{title}\n\n{abstract}"
        return title or abstract
    raise ValueError(
        f"Unknown mode {mode!r}. Use 'title', 'abstract' or 'title_abstract'."
    )
