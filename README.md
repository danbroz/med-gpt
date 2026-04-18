# OpenAlex Medical embedder + GPU FAISS index (RunPod B200, monthly-incremental)

Two pipelines, both running on **NVIDIA B200** (Blackwell, 192 GB HBM3e —
RunPod's most powerful GPU as of April 2026), and both **incremental on
the OpenAlex monthly snapshot**:

| # | Pipeline | Entry point | What it does |
|---|---|---|---|
| 1 | **Embed** | `python -m openalex_med.embed` | Streams the OpenAlex S3 snapshot (public HTTPS), filters to Medicine, reconstructs abstracts from the inverted index, embeds on a B200 + TEI, writes Parquet shards to a local directory. Records every completed snapshot part-file URL in a local `checkpoint.json` so the next monthly run only processes the new `updated_date=*` partitions. |
| 2 | **Index** | `python -m openalex_med.train_runpod` | `rsync`s those Parquet shards (and any existing local index) onto a second B200 and runs `faiss_worker.py`. If no index exists yet → trains a fresh `IndexIVFPQ`. If an index exists → loads it and **adds only the new shards** to the existing index (no retraining). The updated index is `rsync`ed back to your local machine. |

All state for both pipelines lives on the **local filesystem**:
`./embeddings_out/` for parquet shards + checkpoint, and `./index_out/`
for the FAISS index.

## Configure secrets

Copy `.env.example` to `.env` and fill in:

```
RUNPOD_API_KEY=rpa_xxxxxxxxxxxx
```

`.env` is gitignored. Both `embed` and `train_runpod` load it automatically
(via a tiny built-in loader, no `python-dotenv` dependency) and use values
already in `os.environ` in preference to those in the file.

## Install

```bash
pip install -r requirements.txt
```

Local prerequisites for pipeline 2 (orchestrator only — system packages,
not pip):

* `ssh` (openssh-client)
* `rsync`
* an SSH keypair (defaults to `~/.ssh/id_ed25519` then `~/.ssh/id_rsa`,
  or pass `--ssh-public-key`)

The B200 worker installs `faiss-gpu-cu12 pyarrow numpy` itself.

## Pipeline 1 — embed (incremental)

```bash
python -m openalex_med.embed \
    --output ./embeddings_out \
    --mode title_abstract                # 'title' | 'abstract' | 'title_abstract'
```

* On startup, the script reads the local `./embeddings_out/checkpoint.json`
  (if any). The checkpoint lists every OpenAlex snapshot part-file URL
  already processed.
* It re-fetches the OpenAlex Works manifest. The set of part-file URLs in
  the manifest minus the completed set is **exactly the diff** since the
  last run — typically one or two new `updated_date=*` partitions when a
  new monthly snapshot drops.
* For each completed shard it rewrites `checkpoint.json` atomically.

So running it monthly is a no-op unless OpenAlex has shipped new partitions,
and even then it only embeds the newly-added works.

Local output:

```
./embeddings_out/
  checkpoint.json        # {next_shard, completed_parts: [s3 part urls...]}
  manifest.json          # snapshot of the OpenAlex manifest used
  openalex_medical_000000.parquet
  openalex_medical_000001.parquet
  ...
```

Each parquet has columns:

```
id, doi, title, language, publication_year, type, text,
embedding (float32[d])         # 1024 for bge-m3
```

## Pipeline 2 — train / update GPU FAISS (incremental)

```bash
python -m openalex_med.train_runpod \
    --shards-dir ./embeddings_out \
    --output     ./index_out \
    --metric ip --normalize          # cosine similarity
    --nlist 65536 --pq-m 64 --pq-nbits 8
```

What the orchestrator + worker do, in order:

1. Orchestrator launches a **B200** pod from
   `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` with
   your SSH public key injected via `$PUBLIC_KEY`.
2. Polls RunPod for the public 22/tcp mapping, waits for SSH, `rsync`s
   `openalex_med/faiss_worker.py` to `/workspace/`, and `rsync`s the
   parquet shards from `--shards-dir` to `/workspace/embeddings/`.
3. If `--output` already contains a previous index, it `rsync`s those
   `openalex_medical.*` files up to `/workspace/index/` so the worker can
   do an incremental update instead of retraining.
4. Over SSH:
   `pip install --no-cache-dir faiss-gpu-cu12 pyarrow numpy && python /workspace/faiss_worker.py train …`
5. The worker:
   1. **If there's no existing index**: builds `IndexIVFPQ`, ships to GPU(s)
      (`index_cpu_to_all_gpus`, `useFloat16`), trains on
      `nlist × train-sample-mult` vectors, adds every shard.
   2. **If an existing index is present**: loads it, computes the set diff
      `on_disk_shards - indexed_shards.json`, adds *only* those new shards
      (the IVF and PQ codebooks stay frozen — incremental updates don't
      retrain).
   3. Appends to `ids.npy`, updates `indexed_shards.json` and `meta.json`.
6. Orchestrator `rsync`s `/workspace/index/` back to your local `--output`,
   terminates the pod (unless `--keep-pod`).

`meta.json` records `last_update_added_shards`, `last_update_added_vectors`
and `last_update_at` so you can audit what changed each month.

### Useful flags (pipeline 2)

| flag | meaning | default |
|------|---------|---------|
| `--shards-dir` | local parquet shards from pipeline 1 | `./embeddings_out` |
| `--output` | local FAISS index dir (round-tripped through the pod) | `./index_out` |
| `--metric` | `ip` (inner product) or `l2` | `ip` |
| `--no-normalize` | skip L2-norm before adding | off |
| `--nlist` | IVF coarse cells; ~√N on first run | `65536` |
| `--pq-m` | PQ sub-quantizers (`0` → `IndexIVFFlat`) | `64` |
| `--pq-nbits` | bits per PQ code | `8` |
| `--force-rebuild` | ignore the existing local index and retrain | off |
| `--gpu-type-id` | RunPod GPU type id | `NVIDIA B200` |
| `--keep-pod` | leave the pod running afterwards | off |

## Start / restart both pipelines

Both commands are idempotent and resumable:

* Pipeline 1 picks up exactly where its `checkpoint.json` left off — re-run
  it any time and it'll only process new parts.
* Pipeline 2 reads the existing index in `./index_out/` (if any) and only
  adds shards that aren't in `indexed_shards.json` — re-run it any time
  and it'll do an incremental update or a no-op.

```bash
cd /home/dan/med-gpt
python -m openalex_med.embed         --output ./embeddings_out && \
python -m openalex_med.train_runpod  --shards-dir ./embeddings_out --output ./index_out
```

To restart from a clean slate, delete the local state first:

```bash
rm -rf ./embeddings_out ./index_out
python -m openalex_med.embed         --output ./embeddings_out && \
python -m openalex_med.train_runpod  --shards-dir ./embeddings_out --output ./index_out
```

### Suggested monthly cron

```bash
# 1st of each month, after OpenAlex's monthly snapshot drops
0 7 1 * *  cd /home/dan/med-gpt && \
  python -m openalex_med.embed --output ./embeddings_out && \
  python -m openalex_med.train_runpod --shards-dir ./embeddings_out --output ./index_out
```

## Loading the trained index

```python
import faiss, numpy as np

index = faiss.read_index("index_out/openalex_medical.faiss")
ids   = np.load("index_out/openalex_medical.ids.npy", allow_pickle=True)

q = ...                             # (k, d) float32, L2-normalised
D, I = index.search(q, 10)
matches = ids[I]                    # OpenAlex work URLs
```

## Layout

```
.env / .env.example     # secrets (gitignored) + template
openalex_med/
  abstract.py           # reverse-index reconstruction + text builder
  snapshot.py           # streams Medical works from OpenAlex public S3 snapshot (HTTPS)
  runpod_pod.py         # RunPod GraphQL: B200 TEI pod + B200 SSH pod
  embed.py              # pipeline 1: stream → embed on B200 → local parquet
  faiss_worker.py       # GPU FAISS trainer/updater (runs on the B200 pod)
  train_runpod.py       # pipeline 2: launch B200 SSH pod, rsync shards + run worker
  dotenv.py             # tiny dependency-free .env loader
tests/
  test_abstract.py
  test_snapshot.py
  test_faiss_worker.py
  test_train_runpod.py
  test_dotenv.py
```

## Tests

```bash
pytest -q
```

Tests are fully offline — no network, no SSH, no FAISS calls.

## Notes

* TEI's `100-1.9` image is the Blackwell-targeted build (sm_100); the
  `hopper-*` tags will not run on a B200.
* `runpod/pytorch:*-cuda12.8.1-*` is the minimum CUDA stack for B200.
* Both pipelines auto-terminate the pod on completion or Ctrl-C. Pass
  `--keep-pod` to keep it around (and `--reuse-pod-id` on the next run).
