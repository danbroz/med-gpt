"""Tests for the tiny .env loader."""

import os
from pathlib import Path

from openalex_med.dotenv import find_dotenv, load_dotenv


def test_find_dotenv_walks_parents(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FOO=bar\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    found = find_dotenv()
    assert found is not None
    assert Path(found).resolve() == (tmp_path / ".env").resolve()


def test_load_basic(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        "FOO=bar\n"
        "QUOTED=\"hello world\"\n"
        "SQUOTED='hi'\n"
        "EMPTY=\n"
        "MISSING_EQ_LINE\n"
    )
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("SQUOTED", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    applied = load_dotenv(p)
    assert applied == {"FOO": "bar", "QUOTED": "hello world",
                       "SQUOTED": "hi", "EMPTY": ""}
    assert os.environ["FOO"] == "bar"
    assert os.environ["QUOTED"] == "hello world"


def test_does_not_override_by_default(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("FOO=fromfile\n")
    monkeypatch.setenv("FOO", "fromenv")
    load_dotenv(p)
    assert os.environ["FOO"] == "fromenv"


def test_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("FOO=fromfile\n")
    monkeypatch.setenv("FOO", "fromenv")
    load_dotenv(p, override=True)
    assert os.environ["FOO"] == "fromfile"


def test_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}
