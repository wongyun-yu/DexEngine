"""Job list → Kubernetes `batch/v1` Job manifests (one YAML document per job, `kubectl apply -f <file>`).

Each Job runs one list entry in the configured container image; the command and environment come from the job,
the image, namespace, volumes and node selection from `configs/cluster/k8s.yaml` (`K8sConfig`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

from dexengine.cluster.jobs import Job, check_unique


@dataclass(frozen=True)
class K8sConfig:
    """`configs/cluster/k8s.yaml`."""

    image: str = ""
    namespace: str = "default"
    name_prefix: str = "dexengine"
    service_account: str = ""
    image_pull_secret: str = ""
    node_selector: dict[str, str] = field(default_factory=dict)
    #: volumes as {name: {"pvc": claim, "mount": path}} or {name: {"host": path, "mount": path}}
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    backoff_limit: int = 0
    ttl_seconds_after_finished: int = 86400


def _k8s_name(prefix: str, name: str) -> str:
    s = re.sub(r"[^a-z0-9-]", "-", f"{prefix}-{name}".lower()).strip("-")
    return s[:63].rstrip("-")


def manifest(job: Job, cfg: K8sConfig) -> dict:
    """The Job manifest (a dict) of one job."""
    if not cfg.image:
        raise ValueError("K8sConfig.image is empty — set it in configs/cluster/k8s.yaml or with --set image=…")
    res = {"cpu": str(job.cpus), "memory": f"{job.mem_gb}Gi"}
    if job.gpus:
        res["nvidia.com/gpu"] = str(job.gpus)
    container = {
        "name": "job",
        "image": cfg.image,
        "command": list(job.cmd),
        "env": [{"name": k, "value": v} for k, v in sorted(job.env.items())],
        "resources": {"requests": dict(res), "limits": dict(res)},
    }
    if job.workdir:
        container["workingDir"] = job.workdir
    pod: dict = {"restartPolicy": "Never", "containers": [container]}
    if cfg.volumes:
        container["volumeMounts"] = [{"name": n, "mountPath": v["mount"]} for n, v in cfg.volumes.items()]
        pod["volumes"] = [
            {
                "name": n,
                **(
                    {"persistentVolumeClaim": {"claimName": v["pvc"]}}
                    if "pvc" in v
                    else {"hostPath": {"path": v["host"]}}
                ),
            }
            for n, v in cfg.volumes.items()
        ]
    if job.prep:  # runs before the command in the same pod (the pod holds its GPU from scheduling)
        init = {k: v for k, v in container.items() if k != "resources"}
        pod["initContainers"] = [
            {
                **init,
                "name": "prep",
                "command": list(job.prep),
                "resources": {"requests": {"cpu": str(job.cpus), "memory": f"{job.mem_gb}Gi"}},
            }
        ]
    if cfg.node_selector:
        pod["nodeSelector"] = dict(cfg.node_selector)
    if cfg.service_account:
        pod["serviceAccountName"] = cfg.service_account
    if cfg.image_pull_secret:
        pod["imagePullSecrets"] = [{"name": cfg.image_pull_secret}]
    spec: dict = {"backoffLimit": cfg.backoff_limit, "template": {"spec": pod}}
    if cfg.ttl_seconds_after_finished:
        spec["ttlSecondsAfterFinished"] = cfg.ttl_seconds_after_finished
    if job.time_min:
        spec["activeDeadlineSeconds"] = job.time_min * 60
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": _k8s_name(cfg.name_prefix, job.name),
            "namespace": cfg.namespace,
            "labels": {"app": cfg.name_prefix},
        },
        "spec": spec,
    }


def render(jobs: list[Job], cfg: K8sConfig | None = None) -> str:
    """Multi-document YAML of the Job manifests of `jobs` (cfg None = the default K8sConfig)."""
    cfg = K8sConfig() if cfg is None else cfg
    jobs = check_unique(list(jobs))
    names = [_k8s_name(cfg.name_prefix, j.name) for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("two jobs map to the same Kubernetes name (names are lower-cased and cut to 63 characters)")
    return yaml.safe_dump_all([manifest(j, cfg) for j in jobs], sort_keys=False)
