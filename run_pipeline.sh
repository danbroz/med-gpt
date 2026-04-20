#!/usr/bin/env bash
# Two-stage RunPod pipeline:
#   1) embed (single GPU, TEI is single-process)
#   2) index (multi-GPU FAISS via index_cpu_to_all_gpus)
# Each stage falls through every CUDA SKU on RunPod top -> bottom until one
# accepts the deploy. With the patched wait_until_ready, wedged hosts now
# bail in ~8 min instead of waiting the full 30 min, so the || chain
# advances quickly when supply is thin.
set -u
cd /home/dan/med-gpt

run_embed () {
  local gpu="$1" img="$2"
  echo ">>> embed on $gpu (image=$img)"
  python -m openalex_med.embed \
      --output ./embeddings_out \
      --gpu-type-id "$gpu" --gpu-count 1 --image "$img" \
      --threads 20 --embed-threads 8
}

run_index () {
  local gpu="$1" cnt="$2"
  echo ">>> index on $gpu x$cnt"
  python -m openalex_med.train_runpod \
      --shards-dir ./embeddings_out --output ./index_out \
      --gpu-type-id "$gpu" --gpu-count "$cnt"
}

TEI_BLACKWELL="ghcr.io/huggingface/text-embeddings-inference:100-1.9"
TEI_HOPPER="ghcr.io/huggingface/text-embeddings-inference:hopper-1.9"
TEI_ADA="ghcr.io/huggingface/text-embeddings-inference:89-1.9"
TEI_AMPERE="ghcr.io/huggingface/text-embeddings-inference:86-1.9"
TEI_TURING="ghcr.io/huggingface/text-embeddings-inference:turing-1.9"

# Pipeline 1: embed
( run_embed "NVIDIA B300 SXM6 AC"                                    "$TEI_BLACKWELL" \
|| run_embed "NVIDIA B200"                                            "$TEI_BLACKWELL" \
|| run_embed "NVIDIA RTX PRO 6000 Blackwell Server Edition"           "$TEI_BLACKWELL" \
|| run_embed "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"      "$TEI_BLACKWELL" \
|| run_embed "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" "$TEI_BLACKWELL" \
|| run_embed "NVIDIA RTX PRO 4500 Blackwell"                          "$TEI_BLACKWELL" \
|| run_embed "NVIDIA GeForce RTX 5090"                                "$TEI_BLACKWELL" \
|| run_embed "NVIDIA H200 NVL"                                        "$TEI_HOPPER"   \
|| run_embed "NVIDIA H200"                                            "$TEI_HOPPER"   \
|| run_embed "NVIDIA H100 NVL"                                        "$TEI_HOPPER"   \
|| run_embed "NVIDIA H100 80GB HBM3"                                  "$TEI_HOPPER"   \
|| run_embed "NVIDIA H100 PCIe"                                       "$TEI_HOPPER"   \
|| run_embed "NVIDIA L40S"                                            "$TEI_ADA"      \
|| run_embed "NVIDIA L40"                                             "$TEI_ADA"      \
|| run_embed "NVIDIA RTX 6000 Ada Generation"                         "$TEI_ADA"      \
|| run_embed "NVIDIA RTX 5000 Ada Generation"                         "$TEI_ADA"      \
|| run_embed "NVIDIA L4"                                              "$TEI_ADA"      \
|| run_embed "NVIDIA A100-SXM4-80GB"                                  "$TEI_AMPERE"   \
|| run_embed "NVIDIA A100 80GB PCIe"                                  "$TEI_AMPERE"   \
|| run_embed "NVIDIA A100-SXM4-40GB"                                  "$TEI_AMPERE"   \
|| run_embed "NVIDIA A40"                                             "$TEI_AMPERE"   \
|| run_embed "NVIDIA RTX A6000"                                       "$TEI_AMPERE"   \
|| run_embed "NVIDIA RTX A5000"                                       "$TEI_AMPERE"   \
|| run_embed "NVIDIA GeForce RTX 3090"                                "$TEI_AMPERE"   \
|| { echo "ERROR: no embed GPU available on RunPod" >&2; exit 1; } ) \
&& \
# Pipeline 2: index
#
# IMPORTANT: faiss-gpu-cu12 wheels (PyPI, latest 1.14.1.post1 as of Apr 2026)
# only ship cubins for sm_70 (Volta) and sm_80 (Ampere) — NO PTX, so they
# cannot run on Hopper (sm_90), Ada (sm_89), or Blackwell (sm_100/120).
# Only sm_8x SKUs go in this fallback chain. See run_train.sh for the full
# story; verify with `cuobjdump --list-elf` on the installed faiss .so.
( run_index "NVIDIA A100-SXM4-80GB"                                  4  \
|| run_index "NVIDIA A100 80GB PCIe"                                  4  \
|| run_index "NVIDIA A100-SXM4-40GB"                                  4  \
|| run_index "NVIDIA RTX A6000"                                       4  \
|| run_index "NVIDIA A40"                                             4  \
|| run_index "NVIDIA RTX A5000"                                       4  \
|| run_index "NVIDIA GeForce RTX 3090"                                4  \
|| { echo "ERROR: no Ampere index GPU available on RunPod" >&2; exit 1; } )
