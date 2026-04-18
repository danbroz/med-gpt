"""Tests for snapshot helpers (no network access)."""

from openalex_med.snapshot import (
    MEDICINE_CONCEPT_ID,
    is_medical_work,
    manifest_part_urls,
    s3_to_https,
)


def test_s3_to_https():
    assert s3_to_https("s3://openalex/data/works/manifest") == \
        "https://openalex.s3.amazonaws.com/data/works/manifest"
    assert s3_to_https(
        "s3://openalex/data/works/updated_date=2024-01-01/part_000.gz"
    ) == (
        "https://openalex.s3.amazonaws.com/"
        "data/works/updated_date=2024-01-01/part_000.gz"
    )
    assert s3_to_https("https://example.com/x") == "https://example.com/x"


def test_is_medical_work_full_url():
    work = {"concepts": [
        {"id": "https://openalex.org/C71924100", "display_name": "Medicine"},
        {"id": "https://openalex.org/C12345", "display_name": "Other"},
    ]}
    assert is_medical_work(work)


def test_is_medical_work_bare_id():
    assert is_medical_work({"concepts": [{"id": MEDICINE_CONCEPT_ID}]})


def test_not_medical():
    work = {"concepts": [{"id": "https://openalex.org/C99999"}]}
    assert not is_medical_work(work)


def test_no_concepts():
    assert not is_medical_work({})
    assert not is_medical_work({"concepts": None})
    assert not is_medical_work({"concepts": []})


def test_manifest_part_urls_skips_malformed():
    manifest = {
        "entries": [
            {"url": "s3://openalex/data/works/updated_date=2024-01-01/part_000.gz"},
            {"url": "s3://openalex/data/works/updated_date=2024-01-02/part_000.gz"},
            {},
        ]
    }
    urls = manifest_part_urls(manifest)
    assert len(urls) == 2
    assert all(u.startswith("https://openalex.s3.amazonaws.com/") for u in urls)
