"""GPU FAISS index trainer/updater — runs *on the RunPod B200 pod*.

This script is intentionally self-contained (no `openalex_med` imports) so we
can rsync just this single file onto the pod and run it.

Two modes
---------
1. **Initial training** (no existing index in ``--output-dir``):
     * Walk every ``*.parquet`` shard
     * Train an ``IndexIVFPQ`` on a sample of vectors
     * Add every vector
     * Persist ``openalex_medical.faiss``, ``openalex_medical.ids.npy``,
       ``openalex_medical.indexed_shards.json``,
       ``openalex_medical.meta.json``

2. **Incremental update** (existing index found in ``--output-dir``):
     * Read the existing index, ids and ``indexed_shards.json``
     * Identify shards in ``--shards-dir`` whose filename is *not* in
       ``indexed_shards.json``
     * Add their vectors to the loaded index (no retraining — IVF + PQ
       codebooks stay fixed)
     * Append to ``ids.npy`` and write everything back

This pairs perfectly with the embedder's monthly-diff behaviour: the embed
pipeline only writes new shards for new ``updated_date=*`` partitions, and
those new shards are exactly what this worker picks up next time.

Install on the pod (the orchestrator does this automatically)::

    pip install --no-cache-dir faiss-gpu-cu12 pyarrow numpy
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

log = logging.getLogger("faiss_worker")

INDEX_NAME = "openalex_medical.faiss"
IDS_NAME = "openalex_medical.ids.npy"
INDEXED_NAME = "openalex_medical.indexed_shards.json"
META_NAME = "openalex_medical.meta.json"


# =============================================================== parquet IO
def list_shards(shards_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No .parquet shards found in {shards_dir!r}")
    return paths


def iter_vectors(shards: list[str], *, batch_rows: int = 100_000):
    """Yield ``(ids: list[str], vecs: np.ndarray[float32])`` per batch.

    Uses Arrow's flat-buffer reshape (``ListArray.values.to_numpy``) instead
    of ``np.stack`` over Python objects. Empirically ~66,000× faster on
    1024-d embedding shards (0.4 ms vs 26 s for a 65 K row batch), which
    matters a lot when adding ~310 M vectors to a FAISS index.
    """
    for path in shards:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=batch_rows,
            columns=["id", "embedding"],
        ):
            ids = batch.column("id").to_pylist()
            emb = batch.column("embedding")
            n = len(emb)
            if n == 0:
                yield ids, np.empty((0, 0), dtype="float32")
                continue
            # ListArray.values is the flat 1-D arrow array of all child
            # values (n * dim float32s). zero_copy_only=False because Arrow
            # may need to materialize a contiguous buffer; in practice this
            # is still essentially zero-copy for primitive children.
            flat = emb.values.to_numpy(zero_copy_only=False)
            dim = flat.size // n
            vecs = np.ascontiguousarray(
                flat.reshape(n, dim), dtype="float32"
            )
            yield ids, vecs


def detect_dim(shards: list[str]) -> int:
    pf = pq.ParquetFile(shards[0])
    for batch in pf.iter_batches(batch_size=1, columns=["embedding"]):
        v = np.asarray(batch.column("embedding").to_pylist()[0])
        return int(v.shape[0])
    raise RuntimeError("Could not determine embedding dimension from shards")


# ========================================================== state file helpers
def load_indexed_shards(out_dir: Path) -> dict:
    p = out_dir / INDEXED_NAME
    if not p.exists():
        return {"shards": [], "n_vectors": 0}
    return json.loads(p.read_text())


def save_indexed_shards(out_dir: Path, state: dict) -> None:
    (out_dir / INDEXED_NAME).write_text(json.dumps(state, indent=2))


def load_meta(out_dir: Path) -> dict | None:
    p = out_dir / META_NAME
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_meta(out_dir: Path, meta: dict) -> None:
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))


# ============================================================== FAISS helpers
def _build_index(*, dim, nlist, pq_m, pq_nbits, metric):
    import faiss
    metric_type = (
        faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    )
    quantizer = (
        faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    )
    if pq_m and pq_m > 0:
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_nbits)
    else:
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.metric_type = metric_type
    return index


def _to_gpu(cpu_index):
    import faiss
    n_gpus = faiss.get_num_gpus()
    if n_gpus < 1:
        raise RuntimeError(
            "FAISS reports 0 GPUs — install faiss-gpu-cu12 and verify CUDA."
        )
    co = faiss.GpuMultipleClonerOptions()
    co.shard = n_gpus > 1
    co.useFloat16 = True
    if n_gpus == 1:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_index, co), n_gpus
    return faiss.index_cpu_to_all_gpus(cpu_index, co), n_gpus


def _add_shards_to_gpu_index(
    gpu_index,
    shards_to_add: list[str],
    *,
    normalize: bool,
    add_batch: int,
):
    import faiss
    all_ids: list[str] = []
    n_added = 0
    t0 = time.time()
    add_buf: list[np.ndarray] = []
    add_count = 0
    for ids, vecs in iter_vectors(shards_to_add):
        if normalize:
            faiss.normalize_L2(vecs)
        add_buf.append(vecs)
        add_count += vecs.shape[0]
        all_ids.extend(ids)
        if add_count >= add_batch:
            chunk = np.ascontiguousarray(
                np.concatenate(add_buf, axis=0), dtype="float32"
            )
            gpu_index.add(chunk)
            n_added += chunk.shape[0]
            add_buf.clear()
            add_count = 0
            log.info("Added %d vectors so far (%.0f vec/s)",
                     n_added, n_added / max(1.0, time.time() - t0))
    if add_buf:
        chunk = np.ascontiguousarray(
            np.concatenate(add_buf, axis=0), dtype="float32"
        )
        gpu_index.add(chunk)
        n_added += chunk.shape[0]
    log.info("Done adding %d vectors in %.1fs (%.0f vec/s)",
             n_added, time.time() - t0,
             n_added / max(1.0, time.time() - t0))
    return all_ids, n_added


# ================================================================ main entry
def train_or_update(args: argparse.Namespace) -> None:
    import faiss

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.shards_dir).mkdir(parents=True, exist_ok=True)

    # ---- (1) figure out what to do
    shards = list_shards(args.shards_dir)
    log.info("Found %d shard files in %s", len(shards), args.shards_dir)

    state = load_indexed_shards(out_dir)
    already = set(state["shards"])
    new_shards = [s for s in shards if Path(s).name not in already]
    has_existing_index = (out_dir / INDEX_NAME).exists() and not args.force_rebuild

    if has_existing_index and not new_shards:
        log.info("Existing index already covers every shard — nothing to do.")
        return

    dim = detect_dim(shards)
    log.info("Embedding dim = %d", dim)

    if has_existing_index:
        log.info("=== INCREMENTAL UPDATE ===")
        cpu_index = faiss.read_index(str(out_dir / INDEX_NAME))
        existing_ids = list(
            np.load(out_dir / IDS_NAME, allow_pickle=True)
        )
        log.info(
            "Loaded existing index: ntotal=%d, ids=%d, %d new shards to add",
            cpu_index.ntotal, len(existing_ids), len(new_shards),
        )
        if cpu_index.d != dim:
            raise RuntimeError(
                f"Dimension mismatch: existing index d={cpu_index.d} but "
                f"new shards d={dim}. Pass --force-rebuild or check the model."
            )
        gpu_index, n_gpus = _to_gpu(cpu_index)
        log.info("Pushed existing index to %d GPU(s)", n_gpus)
        new_ids, n_added = _add_shards_to_gpu_index(
            gpu_index, new_shards,
            normalize=args.normalize, add_batch=args.add_batch,
        )
        all_ids = existing_ids + new_ids
        n_total = cpu_index.ntotal + n_added  # before gpu→cpu

        meta = load_meta(out_dir) or {}
        meta.update(
            {
                "dim": dim,
                "n_vectors": n_total,
                "metric": meta.get("metric", args.metric),
                "normalized": meta.get("normalized", bool(args.normalize)),
                "shard_count": len(shards),
                "last_update_added_shards": [Path(s).name for s in new_shards],
                "last_update_added_vectors": n_added,
                "last_update_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
            }
        )
    else:
        log.info("=== INITIAL TRAINING ===")
        cpu_index = _build_index(
            dim=dim, nlist=args.nlist, pq_m=args.pq_m,
            pq_nbits=args.pq_nbits, metric=args.metric,
        )
        gpu_index, n_gpus = _to_gpu(cpu_index)
        log.info("Built fresh index on %d GPU(s)", n_gpus)

        target_train = max(args.nlist * args.train_sample_mult,
                           args.nlist * 39)
        # Pre-allocate the training matrix so we never build a giant
        # intermediate list of per-batch arrays and then concatenate them.
        # np.concatenate of ~100 list-array-backed views can take many
        # minutes on this hardware due to allocator/NUMA effects, even
        # though the total data is only ~17 GB.
        log.info(
            "Pre-allocating train_xb shape=(%d, %d) dtype=float32 (~%.1f GB)",
            target_train, dim, target_train * dim * 4 / 1e9,
        )
        train_xb = np.empty((target_train, dim), dtype="float32")
        filled = 0
        sample_t0 = time.time()
        last_log = sample_t0
        for ids, vecs in iter_vectors(shards):
            if args.normalize:
                faiss.normalize_L2(vecs)
            take = min(target_train - filled, vecs.shape[0])
            train_xb[filled:filled + take] = vecs[:take]
            filled += take
            now = time.time()
            if now - last_log >= 10.0:
                log.info(
                    "  …filled %d / %d (%.1f%%) in %.1fs",
                    filled, target_train, 100.0 * filled / target_train,
                    now - sample_t0,
                )
                last_log = now
            if filled >= target_train:
                break
        log.info(
            "Filled train_xb (%d rows) in %.1fs",
            filled, time.time() - sample_t0,
        )

        # Enable FAISS verbose so we can see k-means iterations.
        try:
            gpu_index.verbose = True
            if hasattr(gpu_index, "cp"):
                gpu_index.cp.verbose = True
            if hasattr(gpu_index, "pq"):
                gpu_index.pq.verbose = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not enable verbose on gpu_index: %s", exc)

        log.info("Calling gpu_index.train() on %d vectors …", train_xb.shape[0])
        t0 = time.time()
        gpu_index.train(train_xb)
        log.info("Trained in %.1fs", time.time() - t0)
        del train_xb

        all_ids, n_added = _add_shards_to_gpu_index(
            gpu_index, shards,
            normalize=args.normalize, add_batch=args.add_batch,
        )
        n_total = n_added

        meta = {
            "dim": dim,
            "n_vectors": n_total,
            "nlist": args.nlist,
            "pq_m": args.pq_m,
            "pq_nbits": args.pq_nbits,
            "metric": args.metric,
            "normalized": bool(args.normalize),
            "shard_count": len(shards),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # ---- (3) persist back to disk
    cpu_final = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_final, str(out_dir / INDEX_NAME))
    np.save(out_dir / IDS_NAME, np.asarray(all_ids, dtype=object),
            allow_pickle=True)

    state = {
        "shards": sorted({*already, *(Path(s).name for s in new_shards)})
                  if has_existing_index
                  else sorted(Path(s).name for s in shards),
        "n_vectors": n_total,
    }
    save_indexed_shards(out_dir, state)
    save_meta(out_dir, meta)

    log.info(
        "Wrote %s (ntotal=%d), %s, %s, %s",
        INDEX_NAME, n_total, IDS_NAME, INDEXED_NAME, META_NAME,
    )


# ===================================================================== CLI
def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train or incrementally update a GPU FAISS index over "
                    "OpenAlex Medical embeddings."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="Train (initial) or update (incremental).")
    t.add_argument("--shards-dir", required=True)
    t.add_argument("--output-dir", required=True)
    t.add_argument("--metric", choices=["ip", "l2"], default="ip")
    t.add_argument("--normalize", action="store_true",
                   help="L2-normalize vectors before adding (cosine sim).")
    t.add_argument("--nlist", type=int, default=65_536,
                   help="(Initial training only) IVF coarse cells.")
    t.add_argument("--pq-m", type=int, default=64,
                   help="(Initial training only) PQ sub-quantizers. 0=Flat.")
    t.add_argument("--pq-nbits", type=int, default=8,
                   help="(Initial training only) bits per PQ code.")
    t.add_argument("--train-sample-mult", type=int, default=64)
    t.add_argument("--add-batch", type=int, default=100_000)
    t.add_argument("--force-rebuild", action="store_true",
                   help="Ignore existing index and retrain from scratch.")
    t.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.cmd == "train":
        train_or_update(args)
    else:
        print(f"Unknown command {args.cmd!r}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
