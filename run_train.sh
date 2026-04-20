#!/usr/bin/env bash
# Train-only retry: embed already finished (1788 shards, 333 GB).
# Try every Ampere-major SKU on RunPod top -> bottom until one accepts deploy.
#
# IMPORTANT: faiss-gpu-cu12 wheels (PyPI, latest 1.14.1.post1 as of Apr 2026)
# only ship cubins for sm_70 (Volta) and sm_80 (Ampere) — NO PTX, so they
# cannot run on Hopper (sm_90), Ada (sm_89), or Blackwell (sm_100/120).
# Putting Hopper/Blackwell GPUs in the fallback chain wastes time + money,
# because we'd succeed in deploying them and then crash with
# "CUDA error 209: no kernel image is available for execution on the device"
# during the very first faiss GPU op. Only sm_8x SKUs are listed here.
#
# --container-disk-gb 500 because the default 200 GB can't hold 333 GB of
# parquet shards + the resulting FAISS index.
set -u
cd /home/dan/med-gpt

run_index () {
  local gpu="$1" cnt="$2"
  echo ">>> index on $gpu x$cnt"
  python -m openalex_med.train_runpod \
      --shards-dir ./embeddings_out --output ./index_out \
      --gpu-type-id "$gpu" --gpu-count "$cnt" \
      --container-disk-gb 500
}

( run_index "NVIDIA A100-SXM4-80GB"                                  4  \
|| run_index "NVIDIA A100 80GB PCIe"                                  4  \
|| run_index "NVIDIA A100-SXM4-40GB"                                  4  \
|| run_index "NVIDIA RTX A6000"                                       4  \
|| run_index "NVIDIA A40"                                             4  \
|| run_index "NVIDIA RTX A5000"                                       4  \
|| run_index "NVIDIA GeForce RTX 3090"                                4  \
|| { echo "ERROR: no Ampere index GPU available on RunPod" >&2; exit 1; } )
