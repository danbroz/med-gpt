"""Tests for the FAISS worker's parquet-reading helpers.

These tests don't require `faiss` (we only exercise the IO functions, not the
training path). They write a tiny Parquet shard with the same schema as
`openalex_med.embed` produces, then verify `iter_vectors` / `detect_dim`.
"""

from pathlib import Path

import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
np = pytest.importorskip("numpy")

from openalex_med.faiss_worker import (  # noqa: E402
    detect_dim,
    iter_vectors,
    list_shards,
    load_indexed_shards,
    save_indexed_shards,
)


def _write_shard(path: Path, n: int, dim: int) -> None:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype("float32")
    table = pa.table(
        {
            "id": [f"https://openalex.org/W{i}" for i in range(n)],
            "embedding": list(vecs),
        }
    )
    pq.write_table(table, path)


def test_list_shards_sorted(tmp_path):
    (tmp_path / "openalex_medical_000001.parquet").touch()
    (tmp_path / "openalex_medical_000000.parquet").touch()
    paths = list_shards(str(tmp_path))
    assert [Path(p).name for p in paths] == [
        "openalex_medical_000000.parquet",
        "openalex_medical_000001.parquet",
    ]


def test_list_shards_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_shards(str(tmp_path))


def test_detect_dim(tmp_path):
    _write_shard(tmp_path / "s_000000.parquet", n=3, dim=64)
    assert detect_dim(list_shards(str(tmp_path))) == 64


def test_indexed_shards_roundtrip(tmp_path):
    assert load_indexed_shards(tmp_path) == {"shards": [], "n_vectors": 0}
    save_indexed_shards(
        tmp_path,
        {"shards": ["openalex_medical_000000.parquet"], "n_vectors": 25_000},
    )
    state = load_indexed_shards(tmp_path)
    assert state["shards"] == ["openalex_medical_000000.parquet"]
    assert state["n_vectors"] == 25_000


def test_incremental_shard_diff(tmp_path):
    """Simulate the union semantics used by faiss_worker."""
    state = {
        "shards": [
            "openalex_medical_000000.parquet",
            "openalex_medical_000001.parquet",
        ],
        "n_vectors": 50_000,
    }
    save_indexed_shards(tmp_path, state)
    on_disk = [
        "openalex_medical_000000.parquet",
        "openalex_medical_000001.parquet",
        "openalex_medical_000002.parquet",
        "openalex_medical_000003.parquet",
    ]
    already = set(load_indexed_shards(tmp_path)["shards"])
    new_shards = [s for s in on_disk if s not in already]
    assert new_shards == [
        "openalex_medical_000002.parquet",
        "openalex_medical_000003.parquet",
    ]


def test_iter_vectors_yields_correct_shapes(tmp_path):
    _write_shard(tmp_path / "s_000000.parquet", n=10, dim=8)
    _write_shard(tmp_path / "s_000001.parquet", n=5, dim=8)

    total = 0
    seen_ids = []
    for ids, vecs in iter_vectors(list_shards(str(tmp_path)), batch_rows=4):
        assert vecs.dtype == np.float32
        assert vecs.shape[1] == 8
        assert len(ids) == vecs.shape[0]
        seen_ids.extend(ids)
        total += vecs.shape[0]
    assert total == 15
    # Ids preserved from the parquet
    assert seen_ids[0] == "https://openalex.org/W0"
