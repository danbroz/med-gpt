"""MCP server: medical-paper retrieval over an 85 M-vector OpenAlex FAISS index.

Exposes a single MCP tool, ``search_medical_papers``, that:

1. Embeds the user query with the same model the index was built against
   (``BAAI/bge-m3``, 1024-d, L2-normalized, inner-product search).
2. Runs an IVF-PQ k-NN search against the local FAISS index
   (``openalex_medical.faiss`` + ``openalex_medical.ids.npy``) at k=50 by
   default.
3. Hydrates each hit through the OpenAlex API and returns
   ``title``, ``abstract`` (un-inverted from
   ``abstract_inverted_index``), ``publication_year``, ``doi`` and ``url``.

Transport is **Streamable HTTP** so the same process serves both the MCP
endpoint and a tiny ``/healthz`` for Cloud Run health checks.

Environment variables
---------------------
INDEX_DIR
    Directory containing ``openalex_medical.faiss``,
    ``openalex_medical.ids.npy`` and ``openalex_medical.meta.json``.
    Default: ``/data/index``  (intended for a GCS-FUSE volume mount).
EMBED_MODEL
    HuggingFace model id used for query embedding.
    MUST match the model the index was built with. Default: ``BAAI/bge-m3``.
NPROBE
    IVF probe count. Higher = more accurate, slower. Default: 32.
DEFAULT_K
    Default neighbour count if the caller doesn't pass one. Default: 50.
OPENALEX_MAILTO
    Email address sent to the OpenAlex API for the polite pool
    (https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication).
    Strongly recommended.
PORT
    HTTP port. Cloud Run sets this automatically. Default: 8080.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import threading
import time
from typing import Any
from urllib.parse import quote

import faiss
import httpx
import numpy as np
import uvicorn
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .abstract import reconstruct_abstract

log = logging.getLogger("medgpt.serve")
# basicConfig insists on UPPERCASE level names; tolerate either case so that
# Cloud Run env vars like LOG_LEVEL=info don't blow up the container at
# import time (which otherwise crashes before uvicorn even binds the port,
# making Cloud Run's TCP startup probe fail with no useful diagnostic).
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

INDEX_DIR = os.environ.get("INDEX_DIR", "/data/index")
# Optional local cache directory. Cloud Run gen2's /tmp is a memory-backed
# tmpfs that counts toward the container's memory limit but offers true
# random-access I/O; FAISS index reads otherwise become catastrophically
# slow against GCSFuse. Set to "" to disable local caching.
LOCAL_CACHE_DIR = os.environ.get("LOCAL_CACHE_DIR", "/tmp/index")
# If set (e.g. ``medgpt-danbroz-com-index``), download the index files
# from this GCS bucket to LOCAL_CACHE_DIR at startup using parallel
# chunked transfers. Bypasses GCSFuse entirely, which on Cloud Run can
# stall single-stream reads of multi-GB files for many minutes.
INDEX_GCS_BUCKET = os.environ.get("INDEX_GCS_BUCKET", "")
INDEX_GCS_PREFIX = os.environ.get("INDEX_GCS_PREFIX", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
NPROBE = int(os.environ.get("NPROBE", "32"))
DEFAULT_K = int(os.environ.get("DEFAULT_K", "50"))
MAX_K = int(os.environ.get("MAX_K", "200"))
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "")
PORT = int(os.environ.get("PORT", "8080"))

OPENALEX_BASE = "https://api.openalex.org"
# We ask OpenAlex for only the fields we need — it materially reduces
# bandwidth and is friendlier to the polite pool.
OPENALEX_SELECT = (
    "id,doi,title,publication_year,abstract_inverted_index,"
    "primary_location,best_oa_location,open_access"
)
_W_RE = re.compile(r"W\d+$")


# =========================================================================
# State (loaded once at process start)
# =========================================================================

class _State:
    index: faiss.Index | None = None
    ids: np.ndarray | None = None
    model: SentenceTransformer | None = None
    dim: int | None = None
    normalized: bool = True
    metric_inner_product: bool = True
    phase: str = "booting"
    error: str | None = None

S = _State()


def _set_phase(phase: str) -> None:
    S.phase = phase
    log.info("Warmup phase=%s", phase)


def _stage_locally(name: str) -> str:
    """Make ``name`` available on local disk and return its absolute path.

    Strategy:
      * If ``INDEX_GCS_BUCKET`` is set, download directly from GCS using
        parallel sliced transfers (60-200 MB/s).
      * Else fall back to a sequential copy from the GCSFuse mount at
        ``INDEX_DIR`` (slow but works without GCS auth).
    """
    if not LOCAL_CACHE_DIR:
        return os.path.join(INDEX_DIR, name)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    dst = os.path.join(LOCAL_CACHE_DIR, name)

    if INDEX_GCS_BUCKET:
        import concurrent.futures
        import google.auth  # lazy import: only on Cloud Run / GCP
        import requests
        from google.auth.transport.requests import Request

        obj_name = f"{INDEX_GCS_PREFIX}{name}"
        enc_name = quote(obj_name, safe="")
        meta_url = (
            f"https://storage.googleapis.com/storage/v1/b/"
            f"{INDEX_GCS_BUCKET}/o/{enc_name}"
        )
        media_url = (
            f"https://storage.googleapis.com/download/storage/v1/b/"
            f"{INDEX_GCS_BUCKET}/o/{enc_name}?alt=media"
        )

        _set_phase(f"gcs_auth:{name}")
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
        )
        creds.refresh(Request())
        auth_headers = {
            "Authorization": f"Bearer {creds.token}",
            # Avoid transparent gzip so the byte counts match object size.
            "Accept-Encoding": "identity",
        }
        _set_phase(f"gcs_client:{name}")
        log.info("Prepared authenticated GCS headers for %s", name)

        _set_phase(f"gcs_stat:{name}")
        meta_resp = requests.get(meta_url, headers=auth_headers, timeout=30)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        size = int(meta["size"])
        size_gb = size / 1e9

        if os.path.exists(dst) and os.path.getsize(dst) == size:
            log.info("Cache hit for %s (%.2f GB)", name, size_gb)
            return dst

        t0 = time.time()
        _set_phase(f"download:{name}")
        log.info("Parallel-streaming gs://%s/%s (%.2f GB) -> %s",
                 INDEX_GCS_BUCKET, obj_name, size_gb, dst)
        downloaded = 0
        last_log_t = t0
        last_log_bytes = 0
        progress_lock = threading.Lock()
        chunk_bytes = 256 * 1024 * 1024
        workers = 8

        def _maybe_log_progress() -> None:
            nonlocal last_log_t, last_log_bytes
            now = time.time()
            if downloaded != size and now - last_log_t < 10:
                return
            delta_t = max(now - last_log_t, 1e-6)
            delta_bytes = downloaded - last_log_bytes
            pct = 100.0 * downloaded / max(size, 1)
            mbps = delta_bytes / 1e6 / delta_t
            log.info(
                "Download progress %s %.2f/%.2f GB (%.1f%%, %.0f MB/s)",
                name, downloaded / 1e9, size_gb, pct, mbps,
            )
            last_log_t = now
            last_log_bytes = downloaded

        def _download_range(start: int, end: int) -> None:
            nonlocal downloaded
            headers = dict(auth_headers)
            headers["Range"] = f"bytes={start}-{end}"
            with requests.get(
                media_url,
                headers=headers,
                stream=True,
                timeout=(30, 3600),
            ) as resp:
                if resp.status_code != 206:
                    raise RuntimeError(
                        f"Expected HTTP 206 for range {start}-{end}, "
                        f"got {resp.status_code}"
                    )
                with open(dst, "r+b") as fh:
                    fh.seek(start)
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        with progress_lock:
                            downloaded += len(chunk)
                            _maybe_log_progress()

        with open(dst, "wb") as fh:
            fh.truncate(size)

        ranges = [
            (start, min(start + chunk_bytes - 1, size - 1))
            for start in range(0, size, chunk_bytes)
        ]
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as pool:
                futures = [pool.submit(_download_range, s, e) for s, e in ranges]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
        except Exception:
            with contextlib.suppress(OSError):
                os.remove(dst)
            raise
        dt = time.time() - t0
        log.info("Downloaded %s in %.1fs (%.0f MB/s)",
                 name, dt, size_gb * 1000 / max(dt, 1e-6))
        return dst

    # Fallback: copy from the FUSE mount.
    src = os.path.join(INDEX_DIR, name)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        log.info("Cache hit for %s (%.2f GB)", name, os.path.getsize(dst) / 1e9)
        return dst
    size_gb = os.path.getsize(src) / 1e9
    t0 = time.time()
    _set_phase(f"copy_fuse:{name}")
    log.info("Staging %s (%.2f GB) -> %s via FUSE", src, size_gb, dst)
    with open(src, "rb") as r, open(dst, "wb") as w:
        shutil.copyfileobj(r, w, length=16 * 1024 * 1024)
    dt = time.time() - t0
    log.info("Staged %s in %.1fs (%.0f MB/s)",
             name, dt, size_gb * 1000 / max(dt, 1e-6))
    return dst


def _load_index_and_ids() -> None:
    t0 = time.time()
    _set_phase("stage_index")
    faiss_path = _stage_locally("openalex_medical.faiss")
    _set_phase("stage_ids")
    ids_path = _stage_locally("openalex_medical.ids.npy")
    meta_path = os.path.join(INDEX_DIR, "openalex_medical.meta.json")

    _set_phase("load_faiss")
    log.info("Reading FAISS index from %s", faiss_path)
    index = faiss.read_index(faiss_path)
    # IVF-PQ: nprobe is the only knob exposed at query time. Bump it from
    # the FAISS default of 1 — at nlist=65536 you essentially get noise
    # otherwise. 32 is a good balance for 85 M vectors at PQ64.
    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", NPROBE)
    except Exception as exc:
        log.warning("Could not set nprobe=%d: %s", NPROBE, exc)

    _set_phase("load_ids")
    log.info("Loading id mapping from %s (%.1f GB on disk)",
             ids_path, os.path.getsize(ids_path) / 1e9)
    ids = np.load(ids_path, allow_pickle=True)

    if index.ntotal != ids.shape[0]:
        raise RuntimeError(
            f"Index/IDs size mismatch: index.ntotal={index.ntotal} "
            f"vs ids.shape[0]={ids.shape[0]}"
        )

    S.index = index
    S.ids = ids
    S.dim = index.d

    # Honour the meta if present (normalize/metric).
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as fh:
            meta = json.load(fh)
        S.normalized = bool(meta.get("normalized", True))
        S.metric_inner_product = (meta.get("metric", "ip").lower() == "ip")
        log.info("meta: dim=%d nlist=%s pq_m=%s metric=%s normalized=%s "
                 "n_vectors=%s",
                 meta.get("dim"), meta.get("nlist"), meta.get("pq_m"),
                 meta.get("metric"), meta.get("normalized"),
                 meta.get("n_vectors"))

    # FAISS and numpy have already pulled the data into RAM; the staged
    # files on tmpfs are now pure memory pressure. Remove them so we
    # claw back ~9 GB of the container memory budget.
    if LOCAL_CACHE_DIR and faiss_path.startswith(LOCAL_CACHE_DIR):
        for p in (faiss_path, ids_path):
            try:
                os.remove(p)
                log.info("Freed staged file %s", p)
            except OSError as exc:
                log.warning("Could not unlink %s: %s", p, exc)

    log.info("Loaded index ntotal=%d dim=%d in %.1fs",
             index.ntotal, S.dim, time.time() - t0)


def _load_model() -> None:
    t0 = time.time()
    _set_phase("load_model")
    log.info("Loading embedding model %s on CPU", EMBED_MODEL)
    # device='cpu' is forced because Cloud Run (without --gpu) has no CUDA.
    # bge-m3 is ~2.3 GB; loads in ~10 s from a warm HF cache.
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    S.model = model
    _set_phase("ready")
    log.info("Loaded %s in %.1fs (max_seq_length=%s)",
             EMBED_MODEL, time.time() - t0, model.max_seq_length)


# =========================================================================
# OpenAlex hydration
# =========================================================================

def _to_w_id(openalex_id: str) -> str | None:
    """Extract the bare ``W…`` work id from a full OpenAlex URL or id."""
    if not openalex_id:
        return None
    s = openalex_id.rsplit("/", 1)[-1]
    return s if _W_RE.match(s) else None


def _best_url(work: dict[str, Any]) -> str | None:
    """Pick the most useful 'go read this paper' URL.

    Preference order:
      1. open-access landing page (if work is OA)
      2. primary location landing page
      3. DOI URL
      4. canonical openalex id (always present)
    """
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        return oa["oa_url"]
    boa = work.get("best_oa_location") or {}
    if boa.get("landing_page_url"):
        return boa["landing_page_url"]
    if boa.get("pdf_url"):
        return boa["pdf_url"]
    prim = work.get("primary_location") or {}
    if prim.get("landing_page_url"):
        return prim["landing_page_url"]
    if work.get("doi"):
        return work["doi"]  # already https://doi.org/...
    return work.get("id")


async def _fetch_works(
    client: httpx.AsyncClient, w_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Fetch many works in one OpenAlex call via the ``ids.openalex`` filter.

    The API caps ``per-page`` at 200, which fits k=50 trivially; we still
    chunk to be safe in case the caller bumps k.
    """
    out: dict[str, dict[str, Any]] = {}
    if not w_ids:
        return out

    CHUNK = 100  # well under the per-page=200 cap
    for i in range(0, len(w_ids), CHUNK):
        chunk = w_ids[i : i + CHUNK]
        params = {
            "filter": f"ids.openalex:{'|'.join(chunk)}",
            "select": OPENALEX_SELECT,
            "per-page": str(len(chunk)),
        }
        if OPENALEX_MAILTO:
            params["mailto"] = OPENALEX_MAILTO
        r = await client.get(f"{OPENALEX_BASE}/works", params=params,
                             timeout=20.0)
        r.raise_for_status()
        for w in r.json().get("results", []):
            wid = _to_w_id(w.get("id", ""))
            if wid:
                out[wid] = w
    return out


# =========================================================================
# Search
# =========================================================================

def _embed_query(text: str) -> np.ndarray:
    """Embed a single query string; returns shape (1, dim) float32."""
    assert S.model is not None
    # bge-m3 was trained without a special query prefix, so we just embed
    # the raw text. normalize_embeddings keeps cosine == inner product.
    vec = S.model.encode(
        [text],
        normalize_embeddings=S.normalized,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32", copy=False)
    if vec.shape[1] != S.dim:
        raise RuntimeError(
            f"Query embedding dim={vec.shape[1]} != index dim={S.dim}; "
            f"the model {EMBED_MODEL!r} does not match the index."
        )
    return vec


def _knn(query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    assert S.index is not None and S.ids is not None
    scores, idx = S.index.search(query_vec, k)
    return scores[0], idx[0]


# =========================================================================
# MCP wiring
# =========================================================================

mcp = FastMCP(
    "medgpt",
    instructions=(
        "Semantic search over ~85.5 million biomedical / medical papers "
        "from the OpenAlex corpus. Use search_medical_papers to retrieve "
        "the top-k most semantically similar papers to a natural-language "
        "query and get back title, abstract, year, DOI and URL."
    ),
    host="0.0.0.0",
)


@mcp.tool()
async def search_medical_papers(
    query: str,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Search the OpenAlex medical-paper corpus by semantic similarity.

    Args:
        query: Free-text query, in any language supported by BGE-M3
            (English plus 100+ others). Longer, more descriptive queries
            generally retrieve better than 1-2 keyword queries.
        k: Number of papers to return. Default 50, max 200.

    Returns:
        A dict with:
          query: echo of the input
          k: number of hits actually returned
          results: list of papers, each with:
            rank, score, openalex_id, title, abstract, publication_year,
            doi, url
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    k = max(1, min(int(k), MAX_K))
    if S.index is None or S.model is None:
        raise RuntimeError("Server still warming up; index/model not loaded")

    t0 = time.time()
    qvec = _embed_query(query.strip())
    t_embed = time.time() - t0

    t0 = time.time()
    scores, idx = _knn(qvec, k)
    t_search = time.time() - t0

    hit_full_ids = [S.ids[i] for i in idx if 0 <= i < S.ids.shape[0]]
    hit_w_ids = [_to_w_id(x) for x in hit_full_ids]
    hit_w_ids = [x for x in hit_w_ids if x]

    t0 = time.time()
    async with httpx.AsyncClient(http2=True) as client:
        meta = await _fetch_works(client, hit_w_ids)
    t_fetch = time.time() - t0

    results: list[dict[str, Any]] = []
    for rank, (score, full_id) in enumerate(zip(scores, hit_full_ids), 1):
        wid = _to_w_id(full_id)
        work = meta.get(wid or "", {}) if wid else {}
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        results.append({
            "rank": rank,
            "score": float(score),
            "openalex_id": full_id,
            "title": work.get("title"),
            "abstract": abstract,
            "publication_year": work.get("publication_year"),
            "doi": work.get("doi"),
            "url": _best_url(work),
        })

    log.info(
        "search k=%d query=%r embed=%.0fms knn=%.0fms openalex=%.0fms "
        "hits_with_meta=%d/%d",
        k, query[:80], t_embed * 1000, t_search * 1000, t_fetch * 1000,
        sum(1 for r in results if r["title"]), len(results),
    )
    return {"query": query, "k": len(results), "results": results}


# =========================================================================
# HTTP application: MCP streamable-http + /healthz for Cloud Run
# =========================================================================

async def healthz(_request):  # noqa: D401
    """Liveness/readiness probe for Cloud Run."""
    ready = (S.index is not None and S.model is not None)
    return JSONResponse(
        {
            "ok": ready,
            "ntotal": int(S.index.ntotal) if S.index is not None else 0,
            "dim": S.dim,
            "model": EMBED_MODEL,
            "nprobe": NPROBE,
            "phase": S.phase,
            "error": S.error,
        },
        status_code=200 if ready else 503,
    )


def build_app() -> Starlette:
    # FastMCP's generated ASGI app already exposes its own `/mcp` route.
    # Mount it at `/`, not `/mcp`, otherwise the effective endpoint becomes
    # `/mcp/mcp` and clients pointed at `/mcp` get "session terminated".
    mcp_app = mcp.streamable_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # Critical: do NOT await the heavy loads here — uvicorn only
        # accepts TCP connections AFTER lifespan startup completes, and
        # Cloud Run's default startup probe will kill us long before the
        # 6 GB FAISS index + 2.3 GB embedding model finish loading.
        # Instead, spawn the loads as a fire-and-forget background task
        # so the port binds immediately; /healthz returns 503 until both
        # futures complete and 200 once the server is ready.
        def _bg() -> None:
            try:
                _set_phase("warmup_start")
                _load_index_and_ids()
                _load_model()
                log.info("Server is READY")
            except Exception as exc:
                S.error = f"{type(exc).__name__}: {exc}"
                log.exception("Background warm-up failed; healthz will 503")

        warmup_thread = threading.Thread(
            target=_bg,
            name="medgpt-warmup",
            daemon=True,
        )
        warmup_thread.start()
        # Chain into FastMCP's own lifespan so its session manager,
        # streamable-http transport bookkeeping, etc. all start/stop
        # correctly. Starlette 1.0 dropped on_startup/on_shutdown so this
        # nested lifespan is the only supported way to compose them.
        async with mcp_app.router.lifespan_context(mcp_app):
            try:
                yield
            finally:
                # Daemon thread exits with the process; no coordinated
                # shutdown is needed here.
                pass

    # NOTE: do NOT use the path "/healthz" here — Google's frontend appears
    # to intercept that exact path and return its own 404 before requests
    # ever reach the container, even though every other route works fine.
    # "/health" goes straight through.
    app = Starlette(
        routes=[
            Route("/health", healthz, methods=["GET"]),
            Route("/", healthz, methods=["GET"]),
            Mount("/", app=mcp_app),
        ],
        lifespan=lifespan,
    )
    return app


def main() -> None:
    # Tolerate either-cased LOG_LEVEL for the same reason as basicConfig
    # above; uvicorn requires lowercase here.
    uvicorn.run(
        build_app(),
        host="0.0.0.0",
        port=PORT,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
