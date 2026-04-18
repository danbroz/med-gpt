"""Embed OpenAlex Medical works on a RunPod B200, then train a GPU FAISS index.

Two pipelines, both targeting NVIDIA B200 (Blackwell, RunPod's most powerful
GPU as of April 2026), with monthly-diff incremental updates throughout. All
state (parquet shards + checkpoint + FAISS index) lives on the local
filesystem.

* :mod:`openalex_med.embed`         — stream the OpenAlex S3 snapshot, embed
                                       via TEI on a B200, write Parquet shards
                                       locally; ``checkpoint.json`` persists
                                       which snapshot part-files are done so
                                       re-runs only process the new
                                       ``updated_date=*`` partitions.
* :mod:`openalex_med.train_runpod`  — rsync the local Parquet shards onto a
                                       second B200, then run
                                       :mod:`openalex_med.faiss_worker` which
                                       either trains a fresh ``IndexIVFPQ``
                                       *or* loads an existing local index and
                                       adds only the new shards.
"""

__all__ = [
    "abstract",
    "snapshot",
    "runpod_pod",
    "embed",
    "faiss_worker",
    "train_runpod",
    "dotenv",
]
