"""End-to-end embedder for OpenAlex Medical works.

What it does
------------
1. Launches a RunPod **B200** pod running HuggingFace Text Embeddings
   Inference (TEI) with a multilingual embedding model (default:
   ``BAAI/bge-m3``).
2. Streams every OpenAlex work tagged with the Medicine concept
   (``C71924100``), in any language, **directly from the OpenAlex S3
   snapshot** (gzip-compressed JSON Lines partitioned by ``updated_date``).
3. Reconstructs each abstract from its inverted (reverse) index.
4. Sends batches to the pod's ``/embed`` endpoint.
5. Writes Parquet shards (id, text, language, year, vector) to a local
   ``--output`` directory.
6. Checkpoints the set of completed snapshot part files every shard so:
   * after a crash, the next run picks up exactly where it left off, and
   * after the next monthly OpenAlex snapshot drops, the next run only
     processes the new ``updated_date=*`` partitions.

All state (parquet shards + ``checkpoint.json`` + ``manifest.json``) lives
on the local filesystem under ``--output``. No S3 round-trip.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .abstract import build_text
from .dotenv import load_dotenv
from .runpod_pod import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GPU_TYPE_ID,
    DEFAULT_TEI_IMAGE,
    PodHandle,
    RunPodClient,
)
from .snapshot import (
    WORKS_MANIFEST_URL,
    fetch_manifest,
    iter_parts_concurrent,
    manifest_part_urls,
)

DEFAULT_FETCH_THREADS = 20
DEFAULT_EMBED_THREADS = 8

log = logging.getLogger("openalex_med.embed")

CHECKPOINT_NAME = "checkpoint.json"
MANIFEST_NAME = "manifest.json"


# --------------------------------------------------------------------------- IO
def _shard_path(out_dir: Path, shard_idx: int) -> Path:
    return out_dir / f"openalex_medical_{shard_idx:06d}.parquet"


def _write_shard(
    out_dir: Path,
    shard_idx: int,
    rows: list[dict],
    vectors: np.ndarray,
) -> Path:
    table = pa.table(
        {
            "id": [r["id"] for r in rows],
            "doi": [r["doi"] for r in rows],
            "title": [r["title"] for r in rows],
            "language": [r["language"] for r in rows],
            "publication_year": [r["publication_year"] for r in rows],
            "type": [r["type"] for r in rows],
            "text": [r["text"] for r in rows],
            "embedding": list(vectors.astype("float32")),
        }
    )
    path = _shard_path(out_dir, shard_idx)
    pq.write_table(table, path, compression="zstd")
    return path


def _save_checkpoint(
    out_dir: Path,
    *,
    next_shard: int,
    completed_parts: set[str],
) -> Path:
    """Atomically rewrite ``checkpoint.json`` and return its path."""
    target = out_dir / CHECKPOINT_NAME
    tmp = out_dir / (CHECKPOINT_NAME + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "next_shard": next_shard,
                "completed_parts": sorted(completed_parts),
            }
        )
    )
    tmp.replace(target)
    return target


def _load_checkpoint(out_dir: Path) -> tuple[int, set[str]]:
    p = out_dir / CHECKPOINT_NAME
    if not p.exists():
        return 0, set()
    data = json.loads(p.read_text())
    return (
        int(data.get("next_shard", 0)),
        set(data.get("completed_parts") or []),
    )


# ----------------------------------------------------------------- TEI client
def embed_batch(
    pod: PodHandle,
    texts: list[str],
    *,
    session: requests.Session,
    timeout: float = 300.0,
    max_retries: int = 5,
) -> np.ndarray:
    url = pod.embed_url()
    body = {"inputs": texts, "truncate": True}
    attempt = 0
    while True:
        try:
            r = session.post(url, json=body, timeout=timeout)
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return np.asarray(r.json(), dtype="float32")
        except (requests.RequestException, ValueError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = min(60.0, 2.0**attempt)
            log.warning("TEI request failed (%s); retry %d/%d in %.1fs",
                        exc, attempt, max_retries, wait)
            time.sleep(wait)


# ---------------------------------------------------------------- orchestration
def _embed_in_parallel(
    pod: PodHandle,
    texts: list[str],
    *,
    batch_size: int,
    pool: ThreadPoolExecutor,
    tls: threading.local,
) -> np.ndarray:
    """Embed a list of texts via parallel POSTs to the TEI ``/embed`` endpoint.

    The texts are split into ``batch_size`` chunks, each POSTed concurrently
    by ``pool``. TEI batches additional requests internally on the GPU side,
    so this just keeps its inbound queue saturated. Each worker thread reuses
    a single :class:`requests.Session` (HTTP keep-alive).
    """
    if not texts:
        return np.empty((0, 0), dtype="float32")

    def _do(chunk: list[str]) -> np.ndarray:
        sess = getattr(tls, "session", None)
        if sess is None:
            sess = requests.Session()
            tls.session = sess
        return embed_batch(pod, chunk, session=sess)

    chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    futures = [pool.submit(_do, c) for c in chunks]
    parts = [f.result() for f in futures]
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def run(
    *,
    runpod_api_key: str,
    output_dir: str | Path,
    mode: str = "title_abstract",
    model: str = DEFAULT_EMBEDDING_MODEL,
    gpu_type_id: str = DEFAULT_GPU_TYPE_ID,
    gpu_count: int = 1,
    image: str = DEFAULT_TEI_IMAGE,
    cloud_type: str = "SECURE",
    hf_token: str | None = None,
    batch_size: int = 64,
    shard_size: int = 25_000,
    max_works: int | None = None,
    manifest_url: str = WORKS_MANIFEST_URL,
    keep_pod: bool = False,
    reuse_pod_id: str | None = None,
    skip_no_text: bool = True,
    fetch_threads: int = DEFAULT_FETCH_THREADS,
    embed_threads: int = DEFAULT_EMBED_THREADS,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rp = RunPodClient(runpod_api_key)

    if reuse_pod_id:
        pod = PodHandle(
            pod_id=reuse_pod_id,
            api_url=f"https://{reuse_pod_id}-80.proxy.runpod.net",
            gpu_type_id=gpu_type_id,
            image=image,
            model=model,
        )
        log.info("Reusing existing pod %s", reuse_pod_id)
    else:
        pod = rp.deploy_tei_pod(
            model=model, gpu_type_id=gpu_type_id, gpu_count=gpu_count,
            image=image, cloud_type=cloud_type, hf_token=hf_token,
        )

    def _cleanup(*_args):
        if not keep_pod and not reuse_pod_id:
            try:
                rp.terminate_pod(pod.pod_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to terminate pod %s: %s", pod.pod_id, exc)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        rp.wait_until_ready(pod)

        shard_idx, completed_parts = _load_checkpoint(out_dir)
        log.info(
            "Starting at shard_idx=%d, %d parts already completed",
            shard_idx, len(completed_parts),
        )

        sess = requests.Session()
        manifest = fetch_manifest(manifest_url=manifest_url, session=sess)
        (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest))

        part_urls = [
            u for u in manifest_part_urls(manifest) if u not in completed_parts
        ]
        log.info(
            "Snapshot manifest: %d total parts, %d to process (%d already done) "
            "— fetch_threads=%d, embed_threads=%d",
            len(manifest.get("entries", [])),
            len(part_urls), len(completed_parts),
            fetch_threads, embed_threads,
        )

        shard_rows: list[dict] = []
        shard_vecs: list[np.ndarray] = []
        pending_parts: list[str] = []
        n_total = 0
        t0 = time.time()

        embed_tls = threading.local()
        with ThreadPoolExecutor(
            max_workers=embed_threads, thread_name_prefix="embed",
        ) as embed_pool:
            parts_iter = iter_parts_concurrent(
                part_urls, threads=fetch_threads,
            )
            for part_url, works in parts_iter:
                # Build per-part text + row arrays.
                prepared: list[dict] = []
                texts: list[str] = []
                for w in works:
                    text = build_text(w["title"], w["abstract"], mode=mode)
                    if text:
                        prepared.append({**w, "text": text})
                        texts.append(text)
                    elif not skip_no_text:
                        prepared.append({**w, "text": ""})
                        texts.append("")

                if texts:
                    vectors = _embed_in_parallel(
                        pod, texts,
                        batch_size=batch_size,
                        pool=embed_pool, tls=embed_tls,
                    )
                    assert vectors.shape[0] == len(prepared)
                    shard_rows.extend(prepared)
                    shard_vecs.append(vectors)
                    n_total += len(prepared)

                # All rows from this part are now in the shard buffer (or
                # an earlier already-written shard — which is impossible
                # here). Safe to mark it pending: it'll move to
                # completed_parts the next time we write a shard.
                pending_parts.append(part_url)

                if len(shard_rows) >= shard_size:
                    vecs = np.concatenate(shard_vecs, axis=0)
                    path = _write_shard(out_dir, shard_idx, shard_rows, vecs)
                    log.info(
                        "Wrote shard %s (%d rows, %d total, %.1f docs/s)",
                        path.name, len(shard_rows), n_total,
                        n_total / max(1.0, time.time() - t0),
                    )
                    shard_idx += 1
                    shard_rows.clear()
                    shard_vecs.clear()
                    completed_parts.update(pending_parts)
                    pending_parts.clear()
                    _save_checkpoint(
                        out_dir,
                        next_shard=shard_idx,
                        completed_parts=completed_parts,
                    )

                if max_works is not None and n_total >= max_works:
                    log.info("Reached max_works=%d — stopping early.",
                             max_works)
                    break

        if shard_rows:
            vecs = np.concatenate(shard_vecs, axis=0)
            path = _write_shard(out_dir, shard_idx, shard_rows, vecs)
            shard_idx += 1
            log.info("Wrote final shard %s", path.name)

        if pending_parts or shard_rows:
            completed_parts.update(pending_parts)
            _save_checkpoint(
                out_dir,
                next_shard=shard_idx,
                completed_parts=completed_parts,
            )

        log.info(
            "Done. Embedded %d works in %.1fs (%.1f docs/s) — %d parts complete",
            n_total, time.time() - t0,
            n_total / max(1.0, time.time() - t0),
            len(completed_parts),
        )

    finally:
        if not keep_pod and not reuse_pod_id:
            try:
                rp.terminate_pod(pod.pod_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to terminate pod %s: %s", pod.pod_id, exc)


# ------------------------------------------------------------------------- CLI
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed OpenAlex Medical works on a RunPod B200 "
                    "(reads the OpenAlex S3 snapshot; resumable & incremental)."
    )
    p.add_argument("--runpod-api-key",
                   default=os.environ.get("RUNPOD_API_KEY"),
                   help="RunPod API key (or set $RUNPOD_API_KEY / .env).")
    p.add_argument("--output", "-o", default="./embeddings_out",
                   help="Local directory for parquet shards + checkpoint.")
    p.add_argument("--mode",
                   choices=["title", "abstract", "title_abstract"],
                   default="title_abstract")
    p.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--gpu-type-id", default=DEFAULT_GPU_TYPE_ID,
                   help="RunPod GPU type id. Default: NVIDIA B200.")
    p.add_argument("--gpu-count", type=int, default=1,
                   help="GPUs per pod (1..maxGpuCount for the chosen type). "
                        "TEI is single-process / single-GPU, so values >1 "
                        "are wasted unless you front-end multiple replicas.")
    p.add_argument("--image", default=DEFAULT_TEI_IMAGE)
    p.add_argument("--cloud-type", default="SECURE",
                   choices=["SECURE", "COMMUNITY"])
    p.add_argument("--hf-token",
                   default=os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    p.add_argument("--batch-size", type=int, default=64,
                   help="Texts per TEI POST. TEI re-batches server-side.")
    p.add_argument("--shard-size", type=int, default=25_000)
    p.add_argument("--max-works", type=int, default=None)
    p.add_argument("--threads", type=int, default=DEFAULT_FETCH_THREADS,
                   help="Parallel snapshot-part fetchers (default: 20). "
                        "This is the main throughput knob — the OpenAlex "
                        "snapshot streamer is the bottleneck on small GPUs, "
                        "so 20 concurrent gzip parses keeps TEI's queue full.")
    p.add_argument("--embed-threads", type=int, default=DEFAULT_EMBED_THREADS,
                   help="Parallel TEI POSTs per part (default: 8). TEI "
                        "internally batches across requests, so a small "
                        "in-flight count is enough to saturate the GPU.")
    p.add_argument("--manifest-url", default=WORKS_MANIFEST_URL)
    p.add_argument("--keep-pod", action="store_true")
    p.add_argument("--reuse-pod-id", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.runpod_api_key:
        raise SystemExit(
            "Missing RunPod API key (set RUNPOD_API_KEY in .env, env, or pass "
            "--runpod-api-key)."
        )
    run(
        runpod_api_key=args.runpod_api_key,
        output_dir=args.output,
        mode=args.mode,
        model=args.model,
        gpu_type_id=args.gpu_type_id,
        gpu_count=args.gpu_count,
        image=args.image,
        cloud_type=args.cloud_type,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        max_works=args.max_works,
        manifest_url=args.manifest_url,
        keep_pod=args.keep_pod,
        reuse_pod_id=args.reuse_pod_id,
        fetch_threads=args.threads,
        embed_threads=args.embed_threads,
    )


if __name__ == "__main__":
    main()
