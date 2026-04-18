"""Tests for the FAISS-training orchestrator helpers (no SSH / no network)."""

from openalex_med.runpod_pod import SSHEndpoint
from openalex_med.train_runpod import ssh_argv


def test_ssh_argv_has_required_flags():
    ep = SSHEndpoint(pod_id="abc", host="1.2.3.4", port=22042)
    args = ssh_argv(ep, identity_file=None)
    # Required hardening flags
    assert "-o" in args and "StrictHostKeyChecking=no" in args
    assert "-o" in args and "UserKnownHostsFile=/dev/null" in args
    # Port + user@host
    assert "-p" in args
    assert "22042" in args
    assert "root@1.2.3.4" in args


def test_ssh_argv_with_identity():
    ep = SSHEndpoint(pod_id="abc", host="1.2.3.4", port=22042)
    args = ssh_argv(ep, identity_file="/tmp/id_ed25519")
    assert "-i" in args
    assert "/tmp/id_ed25519" in args


def test_ssh_argv_with_extra():
    ep = SSHEndpoint(pod_id="abc", host="1.2.3.4", port=22042)
    args = ssh_argv(ep, identity_file=None, extra=["echo", "hi"])
    # extra args must come last, after the host
    assert args[-2:] == ["echo", "hi"]


def test_ssh_endpoint_as_args():
    ep = SSHEndpoint(pod_id="x", host="h", port=42)
    assert ep.as_args() == ["root@h", "-p", "42"]
    assert ep.as_args(user="ubuntu") == ["ubuntu@h", "-p", "42"]
