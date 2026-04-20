"""Orchestrator: train (or incrementally update) a GPU FAISS index on a B200.

Workflow
--------
1. Launch a B200 pod from a CUDA/PyTorch image with ``sshd`` enabled.
2. Wait for the pod's public 22/tcp mapping and for SSH to accept connections.
3. ``rsync`` the FAISS worker script (``openalex_med/faiss_worker.py``) and
   the local parquet shards in ``--shards-dir`` up to ``/workspace/``.
   If the local ``--output`` already has an existing index, ``rsync`` it up
   too so the worker can do an incremental update instead of retraining.
4. ``ssh`` in and run::

        pip install --no-cache-dir faiss-gpu-cu12 pyarrow numpy
        python /workspace/faiss_worker.py train ...

5. ``rsync`` ``/workspace/index/`` back to the local ``--output`` dir.
6. Terminate the pod (unless ``--keep-pod`` is passed).
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .dotenv import load_dotenv
from .runpod_pod import (
    DEFAULT_GPU_TYPE_ID,
    DEFAULT_TRAIN_IMAGE,
    PodHandle,
    RunPodClient,
    SSHEndpoint,
)

log = logging.getLogger("openalex_med.train_runpod")


# --------------------------------------------------------------------- helpers
def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(
            f"`{name}` not found on PATH — install openssh-client / rsync."
        )
    return path


def ssh_argv(
    endpoint: SSHEndpoint,
    *,
    identity_file: str | None,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=30",
        "-p", str(endpoint.port),
    ]
    if identity_file:
        args += ["-i", identity_file]
    args.append(f"root@{endpoint.host}")
    if extra:
        args += extra
    return args


def wait_for_tcp(host: str, port: int, *, timeout: float = 600.0,
                 poll_every: float = 5.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                log.info("SSH port open on %s:%d after %.0fs",
                         host, port, time.time() - start)
                return
        except OSError:
            time.sleep(poll_every)
    raise TimeoutError(f"{host}:{port} not reachable within {timeout:.0f}s")


def run_local(cmd: list[str], *, env: dict[str, str] | None = None,
              check: bool = True) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, env=env, check=check)


def run_ssh(
    endpoint: SSHEndpoint,
    remote_cmd: str,
    *,
    identity_file: str | None,
) -> None:
    """Run ``remote_cmd`` over SSH."""
    ssh = require_binary("ssh")
    cmd = [ssh, *ssh_argv(endpoint, identity_file=identity_file,
                          extra=[remote_cmd])]
    run_local(cmd)


def rsync_to_pod(
    endpoint: SSHEndpoint,
    local_path: str,
    remote_path: str,
    *,
    identity_file: str | None,
) -> None:
    rsync = require_binary("rsync")
    ssh_args = " ".join(
        shlex.quote(a)
        for a in ["ssh", *ssh_argv(endpoint, identity_file=identity_file)[:-1]]
    )
    cmd = [rsync, "-az", "--info=progress2,stats0",
           "-e", ssh_args, local_path,
           f"root@{endpoint.host}:{remote_path}"]
    run_local(cmd)


def rsync_from_pod(
    endpoint: SSHEndpoint,
    remote_path: str,
    local_path: str,
    *,
    identity_file: str | None,
) -> None:
    rsync = require_binary("rsync")
    ssh_args = " ".join(
        shlex.quote(a)
        for a in ["ssh", *ssh_argv(endpoint, identity_file=identity_file)[:-1]]
    )
    cmd = [rsync, "-az", "--info=progress2,stats0",
           "-e", ssh_args,
           f"root@{endpoint.host}:{remote_path}",
           local_path]
    run_local(cmd)


def load_public_key(path: str | None) -> tuple[str, str | None]:
    if path:
        pub = Path(path).expanduser()
    else:
        pub = None
        for cand in ("~/.ssh/id_ed25519.pub", "~/.ssh/id_rsa.pub"):
            cand_p = Path(cand).expanduser()
            if cand_p.exists():
                pub = cand_p
                break
        if pub is None:
            raise SystemExit(
                "No SSH public key found. Pass --ssh-public-key or generate "
                "one with `ssh-keygen -t ed25519`."
            )
    if not pub.exists():
        raise SystemExit(f"SSH public key not found: {pub}")
    text = pub.read_text().strip()
    private = pub.with_suffix("")
    return text, str(private) if private.exists() else None


# ------------------------------------------------------------------------- run
def run(
    *,
    runpod_api_key: str,
    shards_dir: str | Path,
    output_dir: str | Path,
    gpu_type_id: str = DEFAULT_GPU_TYPE_ID,
    gpu_count: int = 1,
    image: str = DEFAULT_TRAIN_IMAGE,
    cloud_type: str = "SECURE",
    container_disk_gb: int = 200,
    ssh_public_key_path: str | None = None,
    metric: str = "ip",
    normalize: bool = True,
    nlist: int = 65_536,
    pq_m: int = 64,
    pq_nbits: int = 8,
    train_sample_mult: int = 64,
    add_batch: int = 100_000,
    force_rebuild: bool = False,
    keep_pod: bool = False,
    reuse_pod_id: str | None = None,
) -> None:
    if not shards_dir:
        raise SystemExit("--shards-dir must be provided.")
    sd = Path(shards_dir).resolve()
    if not sd.is_dir():
        raise SystemExit(f"--shards-dir not found: {sd}")
    if not any(sd.glob("*.parquet")):
        raise SystemExit(f"No .parquet shards in {sd}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pub_text, identity = load_public_key(ssh_public_key_path)
    log.info(
        "Using SSH public key %s%s",
        ssh_public_key_path or "<auto>",
        f" + identity {identity}" if identity else "",
    )

    rp = RunPodClient(runpod_api_key)

    if reuse_pod_id:
        pod = PodHandle(
            pod_id=reuse_pod_id,
            api_url=f"https://{reuse_pod_id}.runpod.io",
            gpu_type_id=gpu_type_id,
            image=image,
        )
        log.info("Reusing existing pod %s", reuse_pod_id)
    else:
        pod = rp.deploy_ssh_pod(
            public_key=pub_text,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            image=image,
            cloud_type=cloud_type,
            container_disk_gb=container_disk_gb,
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
        endpoint = rp.get_ssh_endpoint(pod.pod_id)
        log.info("SSH endpoint: root@%s -p %d", endpoint.host, endpoint.port)

        wait_for_tcp(endpoint.host, endpoint.port)
        time.sleep(5)  # sshd settle

        # Push the worker script. Always create the workspace layout.
        # The pytorch base image doesn't ship with rsync, so install it first
        # — without it, rsync_to_pod fails with "rsync: command not found"
        # because rsync invokes the remote rsync binary over the SSH transport.
        worker_local = Path(__file__).resolve().parent / "faiss_worker.py"
        run_ssh(
            endpoint,
            "set -e; "
            "mkdir -p /workspace/embeddings /workspace/index; "
            "if ! command -v rsync >/dev/null 2>&1; then "
            "  echo 'installing rsync on pod...'; "
            "  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
            "   apt-get install -y -qq rsync) "
            "  || (apk add --no-cache rsync) "
            "  || (yum install -y rsync) "
            "  || { echo 'failed to install rsync via apt/apk/yum' >&2; exit 1; }; "
            "fi; "
            "rsync --version | head -n1",
            identity_file=identity,
        )
        rsync_to_pod(endpoint, str(worker_local),
                     "/workspace/faiss_worker.py", identity_file=identity)

        # Push parquet shards.
        rsync_to_pod(endpoint, str(sd).rstrip("/") + "/",
                     "/workspace/embeddings/", identity_file=identity)

        # If we already have an index locally, push it up so the worker does
        # an incremental update instead of retraining from scratch.
        if any(output_dir.glob("openalex_medical.*")):
            rsync_to_pod(endpoint,
                         str(output_dir).rstrip("/") + "/",
                         "/workspace/index/", identity_file=identity)

        # Build the worker invocation.
        worker_args = [
            "train",
            "--shards-dir", "/workspace/embeddings",
            "--output-dir", "/workspace/index",
            "--metric", metric,
            "--nlist", str(nlist),
            "--pq-m", str(pq_m),
            "--pq-nbits", str(pq_nbits),
            "--train-sample-mult", str(train_sample_mult),
            "--add-batch", str(add_batch),
        ]
        if normalize:
            worker_args.append("--normalize")
        if force_rebuild:
            worker_args.append("--force-rebuild")
        worker_argv = " ".join(shlex.quote(a) for a in worker_args)

        remote_cmd = (
            "set -euo pipefail; "
            "pip install --no-cache-dir --quiet faiss-gpu-cu12 pyarrow numpy && "
            f"python /workspace/faiss_worker.py {worker_argv}"
        )
        run_ssh(endpoint, remote_cmd, identity_file=identity)

        # Pull the index back to local.
        rsync_from_pod(endpoint, "/workspace/index/",
                       str(output_dir).rstrip("/") + "/",
                       identity_file=identity)
        log.info("Index files written to %s", output_dir)

    finally:
        if not keep_pod and not reuse_pod_id:
            try:
                rp.terminate_pod(pod.pod_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to terminate pod %s: %s", pod.pod_id, exc)


# ------------------------------------------------------------------------- CLI
def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train (or incrementally update) a GPU FAISS index over "
                    "OpenAlex Medical embeddings on a RunPod B200."
    )
    p.add_argument("--runpod-api-key",
                   default=os.environ.get("RUNPOD_API_KEY"))

    p.add_argument("--shards-dir", default="./embeddings_out",
                   help="Local directory of *.parquet shards (produced by "
                        "pipeline 1). Default: ./embeddings_out.")
    p.add_argument("--output", "-o", default="./index_out",
                   help="Local directory to download the trained index into. "
                        "If it already contains an existing index, it gets "
                        "rsynced up so the worker does an incremental update.")

    p.add_argument("--gpu-type-id", default=DEFAULT_GPU_TYPE_ID,
                   help="RunPod GPU type id. Default: NVIDIA B200.")
    p.add_argument("--gpu-count", type=int, default=1,
                   help="GPUs per pod (1..maxGpuCount for the chosen type). "
                        "faiss_worker.py shards the index across all visible "
                        "GPUs via index_cpu_to_all_gpus, so larger values "
                        "really do speed up training and adds.")
    p.add_argument("--image", default=DEFAULT_TRAIN_IMAGE)
    p.add_argument("--cloud-type", default="SECURE",
                   choices=["SECURE", "COMMUNITY"])
    p.add_argument("--container-disk-gb", type=int, default=200)
    p.add_argument("--ssh-public-key", default=None)

    p.add_argument("--metric", choices=["ip", "l2"], default="ip")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--nlist", type=int, default=65_536)
    p.add_argument("--pq-m", type=int, default=64)
    p.add_argument("--pq-nbits", type=int, default=8)
    p.add_argument("--train-sample-mult", type=int, default=64)
    p.add_argument("--add-batch", type=int, default=100_000)
    p.add_argument("--force-rebuild", action="store_true",
                   help="Ignore the existing local index and retrain.")

    p.add_argument("--keep-pod", action="store_true")
    p.add_argument("--reuse-pod-id", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> None:
    load_dotenv()
    args = _parse(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.runpod_api_key:
        raise SystemExit(
            "Missing RunPod API key (set RUNPOD_API_KEY in .env, env, or "
            "pass --runpod-api-key)."
        )

    run(
        runpod_api_key=args.runpod_api_key,
        shards_dir=args.shards_dir,
        output_dir=args.output,
        gpu_type_id=args.gpu_type_id,
        gpu_count=args.gpu_count,
        image=args.image,
        cloud_type=args.cloud_type,
        container_disk_gb=args.container_disk_gb,
        ssh_public_key_path=args.ssh_public_key,
        metric=args.metric,
        normalize=not args.no_normalize,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
        train_sample_mult=args.train_sample_mult,
        add_batch=args.add_batch,
        force_rebuild=args.force_rebuild,
        keep_pod=args.keep_pod,
        reuse_pod_id=args.reuse_pod_id,
    )


if __name__ == "__main__":
    main()
