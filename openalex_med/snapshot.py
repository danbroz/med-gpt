"""Stream OpenAlex Medical works directly from the public S3 snapshot.

The snapshot lives in the unauthenticated bucket ``s3://openalex/`` and is
mirrored at ``https://openalex.s3.amazonaws.com/``. For Works the layout is::

    /data/works/manifest                     (Redshift-format JSON)
    /data/works/updated_date=YYYY-MM-DD/
        part_000.gz   part_001.gz   ...      (gzip-compressed JSON Lines)

Each line in a part file is one full Work object — the same schema the API
returns. We stream each part file over HTTPS, decompress on the fly, and
filter locally to the Medicine concept (``C71924100``).

Why HTTPS instead of boto3?
    * The bucket is public, so no credentials are needed.
    * `requests` + `gzip.GzipFile` lets us stream without buffering whole
      files in memory and avoids a `boto3` dependency.

Checkpointing happens at part-file granularity: once we finish a part, the
caller records its URL in `processed_parts` and never revisits it.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Iterable, Iterator
from urllib.parse import urlparse

import requests

from .abstract import reconstruct_abstract

log = logging.getLogger(__name__)

OPENALEX_BUCKET = "openalex"
OPENALEX_HTTPS_BASE = f"https://{OPENALEX_BUCKET}.s3.amazonaws.com"
WORKS_MANIFEST_URL = f"{OPENALEX_HTTPS_BASE}/data/works/manifest"

MEDICINE_CONCEPT_ID = "C71924100"  # top-level Medicine concept


# --------------------------------------------------------------------- helpers
def s3_to_https(s3_url: str) -> str:
    """Convert ``s3://openalex/...`` to the public HTTPS URL."""
    p = urlparse(s3_url)
    if p.scheme != "s3":
        return s3_url
    bucket = p.netloc
    key = p.path.lstrip("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def is_medical_work(work: dict, concept_id: str = MEDICINE_CONCEPT_ID) -> bool:
    """Return True if a Work is tagged with the Medicine concept.

    OpenAlex stores concept ids as full URLs (e.g.
    ``https://openalex.org/C71924100``); we just check the suffix so the test
    is robust to URL changes.
    """
    suffix = "/" + concept_id
    for c in work.get("concepts") or ():
        cid = c.get("id") or ""
        if cid.endswith(suffix) or cid == concept_id:
            return True
    return False


# --------------------------------------------------------------------- manifest
def fetch_manifest(
    *,
    manifest_url: str = WORKS_MANIFEST_URL,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> dict:
    """Download and parse the Works manifest (Redshift format)."""
    sess = session or requests.Session()
    r = sess.get(manifest_url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def manifest_part_urls(manifest: dict) -> list[str]:
    """Extract every part-file HTTPS URL from a manifest, in manifest order."""
    urls: list[str] = []
    for entry in manifest.get("entries", []):
        url = entry.get("url")
        if not url:
            continue
        urls.append(s3_to_https(url))
    return urls


# ---------------------------------------------------------------- streaming IO
def stream_part(
    url: str,
    *,
    session: requests.Session | None = None,
    chunk_size: int = 1 << 20,  # 1 MiB
    max_retries: int = 5,
    backoff: float = 2.0,
    timeout: float = 120.0,
) -> Iterator[dict]:
    """Stream one ``.gz`` part file and yield decoded JSON objects.

    Decompresses incrementally (never holds the full part in memory) and
    retries the whole part on transient HTTP errors — gzip streams cannot be
    cleanly resumed mid-file, so a fresh GET is the safe option.
    """
    sess = session or requests.Session()
    attempt = 0
    while True:
        try:
            with sess.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                # raw stream → gzip decoder → text reader → JSONL
                resp.raw.decode_content = False  # we handle gzip ourselves
                with gzip.GzipFile(fileobj=resp.raw) as gz:
                    reader = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
                    for line in reader:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            log.warning("Skipping bad JSON line in %s: %s", url, exc)
            return
        except (requests.RequestException, OSError, EOFError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = backoff ** attempt
            log.warning(
                "Part download failed (%s) for %s — retry %d/%d in %.1fs",
                exc, url, attempt, max_retries, wait,
            )
            time.sleep(wait)


# ----------------------------------------------------------- public iterator
def stream_medical_works_from_snapshot(
    *,
    manifest: dict | None = None,
    manifest_url: str = WORKS_MANIFEST_URL,
    skip_parts: Iterable[str] = (),
    session: requests.Session | None = None,
    max_works: int | None = None,
    concept_id: str = MEDICINE_CONCEPT_ID,
) -> Iterator[dict]:
    """Yield Medical works one by one straight from the OpenAlex S3 dump.

    Each yielded dict has::

        id, doi, title, language, publication_year, type,
        abstract,        # reconstructed from abstract_inverted_index, or None
        _part_url,       # part file this work came from
        _part_done,      # True on the LAST work yielded from each part

    Use ``_part_done`` together with ``_part_url`` for checkpointing: persist
    the part URL only once you've finished embedding everything that came
    from it, and pass already-completed URLs back via ``skip_parts`` on the
    next run.
    """
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", "openalex-med-embedder/0.2 (+snapshot)")

    if manifest is None:
        manifest = fetch_manifest(manifest_url=manifest_url, session=sess)

    skip = set(skip_parts)
    part_urls = [u for u in manifest_part_urls(manifest) if u not in skip]
    log.info(
        "Snapshot manifest: %d total parts, %d to process (%d already done)",
        len(manifest.get("entries", [])), len(part_urls), len(skip),
    )

    yielded = 0
    for url in part_urls:
        # Buffer one work so we can mark the LAST yield from each part with
        # _part_done=True (lets the caller checkpoint atomically).
        prev: dict | None = None
        n_in_part = 0
        for work in stream_part(url, session=sess):
            if not is_medical_work(work, concept_id):
                continue
            mapped = {
                "id": work.get("id"),
                "doi": work.get("doi"),
                "title": work.get("title") or work.get("display_name"),
                "language": work.get("language"),
                "publication_year": work.get("publication_year"),
                "type": work.get("type"),
                "abstract": reconstruct_abstract(
                    work.get("abstract_inverted_index")
                ),
                "_part_url": url,
                "_part_done": False,
            }
            if prev is not None:
                yield prev
                yielded += 1
                if max_works is not None and yielded >= max_works:
                    return
            prev = mapped
            n_in_part += 1

        if prev is not None:
            prev["_part_done"] = True
            yield prev
            yielded += 1
            log.info("Finished part %s (%d medical works)", url, n_in_part)
            if max_works is not None and yielded >= max_works:
                return
        else:
            log.info("Finished part %s (0 medical works)", url)
            # Emit a sentinel so the caller can still mark this part done.
            yield {
                "id": None, "doi": None, "title": None, "language": None,
                "publication_year": None, "type": None, "abstract": None,
                "_part_url": url, "_part_done": True, "_sentinel": True,
            }


# ----------------------------------------------------- per-part / concurrent
def _map_work(work: dict) -> dict:
    """Project an OpenAlex Work down to the columns we keep, with abstract."""
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title") or work.get("display_name"),
        "language": work.get("language"),
        "publication_year": work.get("publication_year"),
        "type": work.get("type"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
    }


def medical_works_for_part(
    url: str,
    *,
    session: requests.Session | None = None,
    concept_id: str = MEDICINE_CONCEPT_ID,
) -> list[dict]:
    """Download one snapshot part, return every Medical work in it (mapped).

    This is the unit of work parallelised by :func:`iter_parts_concurrent`.
    """
    rows: list[dict] = []
    for work in stream_part(url, session=session):
        if not is_medical_work(work, concept_id):
            continue
        rows.append(_map_work(work))
    return rows


def iter_parts_concurrent(
    part_urls: Iterable[str],
    *,
    threads: int = 20,
    concept_id: str = MEDICINE_CONCEPT_ID,
    session_factory=None,
) -> Iterator[tuple[str, list[dict]]]:
    """Yield ``(part_url, [medical_works])`` in **completion** order.

    Up to ``threads`` parts are downloaded + parsed + filtered in parallel
    (each in its own thread, with a thread-local :class:`requests.Session`).
    Parts are produced as a single batch per part — this is what lets the
    embed pipeline mark a part "done" atomically (every row from a finished
    part is in hand before we yield it).

    The number of futures in flight at any moment is bounded to
    ``threads * 2``, so we never sit on more than ~2 parts of buffered
    medical works per worker. The yielded order is the order parts *finish*
    (not the manifest order); callers that need stable shard contents should
    not rely on order.
    """
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")

    tls = threading.local()

    def _sess() -> requests.Session:
        s = getattr(tls, "session", None)
        if s is None:
            s = (session_factory() if session_factory else requests.Session())
            s.headers.setdefault(
                "User-Agent", "openalex-med-embedder/0.3 (+snapshot,concurrent)"
            )
            tls.session = s
        return s

    def _do(url: str) -> tuple[str, list[dict]]:
        rows = medical_works_for_part(
            url, session=_sess(), concept_id=concept_id,
        )
        log.info("Finished part %s (%d medical works)", url, len(rows))
        return url, rows

    pool = ThreadPoolExecutor(
        max_workers=threads, thread_name_prefix="snapshot-fetch",
    )
    try:
        urls_iter = iter(part_urls)
        inflight: dict = {}
        prefetch = max(1, threads * 2)
        for _ in range(prefetch):
            try:
                u = next(urls_iter)
            except StopIteration:
                break
            inflight[pool.submit(_do, u)] = u

        while inflight:
            done, _pending = wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut)
                yield fut.result()
                try:
                    u = next(urls_iter)
                    inflight[pool.submit(_do, u)] = u
                except StopIteration:
                    pass
    finally:
        # Cancel any pending submissions then drain workers without waiting
        # on their HTTP downloads (they will still try to finish gracefully).
        pool.shutdown(wait=False, cancel_futures=True)
