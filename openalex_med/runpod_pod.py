"""Manage RunPod GPU pods for the embedding and FAISS-training pipelines.

We talk to RunPod via its GraphQL API (the REST API doesn't yet cover every
deploy option). Two pod flavors are supported:

* :func:`RunPodClient.deploy_tei_pod` — runs HuggingFace Text Embeddings
  Inference (TEI), exposing an OpenAI-compatible HTTP server. Used by
  :mod:`openalex_med.embed`.

* :func:`RunPodClient.deploy_ssh_pod` — runs a generic CUDA/PyTorch image
  with `sshd` and the user's SSH public key pre-installed. Used by
  :mod:`openalex_med.train_runpod` so we can rsync parquet shards in, run
  GPU FAISS, and rsync the resulting index back out.

Both pipelines default to the **NVIDIA B200** (Blackwell, 192 GB HBM3e) —
the most powerful GPU in RunPod's April 2026 lineup.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

# Most powerful GPU on RunPod (April 2026): NVIDIA B200 SXM, 192 GB HBM3e,
# Blackwell architecture (compute capability sm_100).
DEFAULT_GPU_TYPE_ID = "NVIDIA B200"

# TEI image tag for Blackwell (sm_100). The "hopper-*" tags are sm_90 only.
DEFAULT_TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:100-1.9"

# Multilingual default model. BGE-M3 → 1024-d vectors, 100+ languages
# (incl. medical/scientific text), good fit for "any language" OpenAlex Works.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

# Base image for the FAISS-training pod. Comes with CUDA 12.8 (required for
# B200 / Blackwell) and an SSH server that auto-loads $PUBLIC_KEY into
# /root/.ssh/authorized_keys on boot.
DEFAULT_TRAIN_IMAGE = (
    "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
)


@dataclass
class PodHandle:
    """Lightweight reference to a running RunPod pod."""

    pod_id: str
    api_url: str  # e.g. https://abc123-80.proxy.runpod.net (TEI pods)
    gpu_type_id: str
    image: str
    model: str | None = None

    def health_url(self) -> str:
        return f"{self.api_url}/health"

    def embed_url(self) -> str:
        return f"{self.api_url}/embed"

    def openai_embed_url(self) -> str:
        return f"{self.api_url}/v1/embeddings"


@dataclass
class SSHEndpoint:
    """Public IP + port for SSH'ing into a RunPod pod."""

    pod_id: str
    host: str
    port: int

    def as_args(self, user: str = "root") -> list[str]:
        """Return the ssh `[user@host, -p, port]` argv fragment."""
        return [f"{user}@{self.host}", "-p", str(self.port)]


class RunPodClient:
    def __init__(self, api_key: str, *, session: requests.Session | None = None):
        if not api_key:
            raise ValueError("RunPod API key is required")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # ---------------------------------------------------------------- GraphQL
    def _gql(self, query: str, variables: dict | None = None) -> dict:
        resp = self.session.post(
            RUNPOD_GRAPHQL,
            data=json.dumps({"query": query, "variables": variables or {}}),
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload and payload["errors"]:
            raise RuntimeError(f"RunPod GraphQL error: {payload['errors']}")
        return payload["data"]

    # ------------------------------------------------------- TEI (embedding)
    def deploy_tei_pod(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        gpu_type_id: str = DEFAULT_GPU_TYPE_ID,
        gpu_count: int = 1,
        image: str = DEFAULT_TEI_IMAGE,
        name: str = "openalex-med-embedder",
        cloud_type: str = "SECURE",
        container_disk_gb: int = 80,
        volume_gb: int = 0,
        max_batch_tokens: int = 32768,
        max_client_batch_size: int = 256,
        hf_token: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> PodHandle:
        """Launch a TEI pod (B200 by default) and return its proxy handle."""
        docker_args = (
            f"--model-id {model} "
            f"--max-batch-tokens {max_batch_tokens} "
            f"--max-client-batch-size {max_client_batch_size} "
            f"--port 80"
        )
        env = {"MODEL_ID": model}
        if hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        if extra_env:
            env.update(extra_env)
        env_list = [{"key": k, "value": v} for k, v in env.items()]

        mutation = """
        mutation deployPod($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            imageName
            machineId
            desiredStatus
          }
        }
        """
        if gpu_count < 1:
            raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
        variables = {
            "input": {
                "cloudType": cloud_type,
                "gpuCount": gpu_count,
                "gpuTypeId": gpu_type_id,
                "name": name,
                "imageName": image,
                "dockerArgs": docker_args,
                "ports": "80/http",
                "containerDiskInGb": container_disk_gb,
                "volumeInGb": volume_gb,
                "env": env_list,
            }
        }

        data = self._gql(mutation, variables)
        pod = data["podFindAndDeployOnDemand"]
        if not pod or not pod.get("id"):
            raise RuntimeError(
                "RunPod did not return a pod id — likely no capacity for "
                f"GPU type {gpu_type_id!r}. Try a different cloudType/gpu."
            )
        pod_id = pod["id"]
        api_url = f"https://{pod_id}-80.proxy.runpod.net"
        log.info("Deployed TEI pod %s on %s — proxy: %s", pod_id, gpu_type_id, api_url)
        return PodHandle(
            pod_id=pod_id,
            api_url=api_url,
            gpu_type_id=gpu_type_id,
            image=image,
            model=model,
        )

    # --------------------------------------------------- SSH (FAISS training)
    def deploy_ssh_pod(
        self,
        *,
        public_key: str,
        gpu_type_id: str = DEFAULT_GPU_TYPE_ID,
        gpu_count: int = 1,
        image: str = DEFAULT_TRAIN_IMAGE,
        name: str = "openalex-med-faiss",
        cloud_type: str = "SECURE",
        container_disk_gb: int = 200,
        volume_gb: int = 0,
        extra_env: dict[str, str] | None = None,
    ) -> PodHandle:
        """Launch a CUDA pod that exposes SSH on a public TCP port.

        ``runpod/pytorch:*`` base images source ``$PUBLIC_KEY`` into
        ``/root/.ssh/authorized_keys`` and start ``sshd`` automatically.
        We expose ``22/tcp`` so RunPod assigns a public ip+port mapping;
        retrieve it later with :meth:`get_ssh_endpoint`.
        """
        env = {"PUBLIC_KEY": public_key.strip()}
        if extra_env:
            env.update(extra_env)
        env_list = [{"key": k, "value": v} for k, v in env.items()]

        mutation = """
        mutation deployPod($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            imageName
            machineId
            desiredStatus
          }
        }
        """
        if gpu_count < 1:
            raise ValueError(f"gpu_count must be >= 1, got {gpu_count}")
        variables = {
            "input": {
                "cloudType": cloud_type,
                "gpuCount": gpu_count,
                "gpuTypeId": gpu_type_id,
                "name": name,
                "imageName": image,
                # 22/tcp -> RunPod gives us a publicIp + publicPort mapping.
                "ports": "22/tcp",
                "containerDiskInGb": container_disk_gb,
                "volumeInGb": volume_gb,
                "env": env_list,
            }
        }

        data = self._gql(mutation, variables)
        pod = data["podFindAndDeployOnDemand"]
        if not pod or not pod.get("id"):
            raise RuntimeError(
                f"No capacity for GPU type {gpu_type_id!r} in {cloud_type}."
            )
        pod_id = pod["id"]
        log.info("Deployed SSH pod %s on %s", pod_id, gpu_type_id)
        # api_url is irrelevant for SSH pods; keep it for compatibility.
        return PodHandle(
            pod_id=pod_id,
            api_url=f"https://{pod_id}.runpod.io",
            gpu_type_id=gpu_type_id,
            image=image,
        )

    # --------------------------------------------------------------- pod ops
    def get_pod(self, pod_id: str) -> dict:
        query = """
        query pod($input: PodFilter!) {
          pod(input: $input) {
            id
            desiredStatus
            lastStatusChange
            runtime {
              uptimeInSeconds
              ports { ip isIpPublic privatePort publicPort type }
            }
          }
        }
        """
        return self._gql(query, {"input": {"podId": pod_id}})["pod"]

    def stop_pod(self, pod_id: str) -> None:
        mutation = """
        mutation stopPod($input: PodStopInput!) {
          podStop(input: $input) { id desiredStatus }
        }
        """
        self._gql(mutation, {"input": {"podId": pod_id}})
        log.info("Stopped pod %s", pod_id)

    def terminate_pod(self, pod_id: str) -> None:
        mutation = """
        mutation terminatePod($input: PodTerminateInput!) {
          podTerminate(input: $input)
        }
        """
        self._gql(mutation, {"input": {"podId": pod_id}})
        log.info("Terminated pod %s", pod_id)

    # ---------------------------------------------------------------- waiting
    def wait_until_ready(
        self,
        pod: PodHandle,
        *,
        timeout: float = 1800.0,
        poll_every: float = 10.0,
    ) -> None:
        """Block until the TEI server inside the pod responds 200 on /health."""
        start = time.time()
        last_err: Exception | None = None
        while time.time() - start < timeout:
            try:
                r = requests.get(pod.health_url(), timeout=10)
                if r.status_code == 200:
                    log.info(
                        "Pod %s is healthy after %.0fs",
                        pod.pod_id, time.time() - start,
                    )
                    return
                last_err = RuntimeError(f"health returned {r.status_code}")
            except requests.RequestException as exc:
                last_err = exc
            time.sleep(poll_every)
        raise TimeoutError(
            f"Pod {pod.pod_id} did not become healthy in {timeout:.0f}s "
            f"(last error: {last_err})"
        )

    def get_ssh_endpoint(
        self,
        pod_id: str,
        *,
        timeout: float = 600.0,
        poll_every: float = 5.0,
    ) -> SSHEndpoint:
        """Poll ``pod.runtime.ports`` until the public 22/tcp mapping appears."""
        start = time.time()
        while time.time() - start < timeout:
            info = self.get_pod(pod_id)
            ports = (info.get("runtime") or {}).get("ports") or []
            for p in ports:
                if (
                    p.get("privatePort") == 22
                    and p.get("type") == "tcp"
                    and p.get("isIpPublic")
                    and p.get("ip")
                    and p.get("publicPort")
                ):
                    return SSHEndpoint(
                        pod_id=pod_id,
                        host=p["ip"],
                        port=int(p["publicPort"]),
                    )
            time.sleep(poll_every)
        raise TimeoutError(
            f"Pod {pod_id} did not expose a public SSH port in {timeout:.0f}s"
        )
