"""Deploy an SSH-enabled training pod with GPU fallback, and print its info.

Used to spin up a *destination* pod for the pod-to-pod transfer from the
Blackwell pod (whose faiss-gpu wheel can't run on sm_120) to a Hopper /
Ampere pod whose faiss-gpu-cu12 wheel actually works.

Prints a JSON line with:
  pod_id, gpu_type_id, ssh_host, ssh_port

so the calling shell can parse it with `python -c json` or `jq`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from openalex_med.runpod_pod import RunPodClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("deploy_train_pod")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpu-types",
        nargs="+",
        default=[
            "NVIDIA H200",
            "NVIDIA H100 80GB HBM3",
            "NVIDIA H100 PCIe",
            "NVIDIA A100 80GB PCIe",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA RTX A6000",
        ],
        help="Ordered list of GPU type ids to try, first-success wins.",
    )
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--container-disk-gb", type=int, default=500)
    ap.add_argument(
        "--public-key-file",
        default=str(Path.home() / ".ssh" / "id_ed25519.pub"),
    )
    ap.add_argument("--name", default="openalex-med-faiss-hopper")
    ap.add_argument("--cloud-type", default="SECURE")
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY is not set in the environment")

    public_key = Path(args.public_key_file).read_text().strip()

    client = RunPodClient(api_key)

    pod = None
    last_err: Exception | None = None
    for gpu_type in args.gpu_types:
        log.info("Trying gpu_type=%r in cloud=%s", gpu_type, args.cloud_type)
        try:
            pod = client.deploy_ssh_pod(
                public_key=public_key,
                gpu_type_id=gpu_type,
                gpu_count=args.gpu_count,
                container_disk_gb=args.container_disk_gb,
                name=args.name,
                cloud_type=args.cloud_type,
            )
            log.info("Got pod %s on %s", pod.pod_id, gpu_type)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("gpu=%r failed: %s", gpu_type, exc)

    if pod is None:
        sys.exit(f"All GPU types exhausted; last error: {last_err}")

    log.info("Waiting for SSH endpoint on pod %s", pod.pod_id)
    endpoint = client.get_ssh_endpoint(pod.pod_id, timeout=900, poll_every=10)
    log.info(
        "SSH endpoint ready: %s:%s (pod=%s)",
        endpoint.host, endpoint.port, pod.pod_id,
    )

    # Give sshd a few seconds to actually accept connections.
    time.sleep(5)

    print(json.dumps({
        "pod_id": pod.pod_id,
        "gpu_type_id": pod.gpu_type_id,
        "ssh_host": endpoint.host,
        "ssh_port": endpoint.port,
    }))


if __name__ == "__main__":
    main()
