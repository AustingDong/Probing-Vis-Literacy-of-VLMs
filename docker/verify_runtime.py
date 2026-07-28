"""Fail-fast NVIDIA CUDA or AMD ROCm validation for Docker entry points."""

from __future__ import annotations

import os
from pathlib import Path
import pwd
import sys

import torch


WRITABLE_DIRECTORIES = (
    Path("/home/user/.cache/huggingface"),
    Path("/home/user/.cache/torch"),
    Path("/home/user/.cache/matplotlib"),
    Path("/home/user/app/results"),
)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def prepare_directories_and_drop_privileges() -> None:
    """Make mounted volumes writable, then permanently become the app user."""
    if os.geteuid() != 0:
        return

    account = pwd.getpwnam("user")
    supplementary_groups = {group for group in os.getgroups() if group != 0}
    gpu_devices = (Path("/dev/kfd"), *Path("/dev/dri").glob("*"))
    for device in gpu_devices:
        if device.exists():
            device_group = device.stat().st_gid
            if device_group != 0:
                supplementary_groups.add(device_group)

    for directory in WRITABLE_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        for root, directories, files in os.walk(directory):
            os.chown(root, account.pw_uid, account.pw_gid)
            for name in directories:
                os.chown(
                    os.path.join(root, name),
                    account.pw_uid,
                    account.pw_gid,
                    follow_symlinks=False,
                )
            for name in files:
                os.chown(
                    os.path.join(root, name),
                    account.pw_uid,
                    account.pw_gid,
                    follow_symlinks=False,
                )

    # AMD device nodes are commonly owned by the host's video/render groups.
    # Preserve their numeric GIDs even when they do not match names in the
    # container's /etc/group, then permanently drop root.
    os.setgroups(sorted(supplementary_groups))
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)

    if os.geteuid() == 0:
        raise RuntimeError("Failed to drop root privileges.")


def configured_backend() -> str:
    backend = os.getenv("GPU_BACKEND", "auto").strip().lower()
    aliases = {"amd": "rocm", "cuda": "nvidia"}
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "nvidia", "rocm"}:
        raise RuntimeError(
            "GPU_BACKEND must be one of: auto, nvidia, rocm "
            f"(received {backend!r})."
        )
    return backend


def detected_backend() -> str | None:
    if torch.version.hip is not None:
        return "rocm"
    if torch.version.cuda is not None:
        return "nvidia"
    return None


def verify_gpu() -> None:
    # REQUIRE_CUDA remains a fallback for old deployments created before the
    # NVIDIA and AMD entry points were split.
    require_gpu = env_flag("REQUIRE_GPU", env_flag("REQUIRE_CUDA", True))
    expected_backend = configured_backend()
    actual_backend = detected_backend()

    if expected_backend != "auto" and actual_backend != expected_backend:
        expected_runtime = "CUDA" if expected_backend == "nvidia" else "ROCm/HIP"
        installed_runtime = actual_backend or "CPU-only"
        raise RuntimeError(
            f"This image expects {expected_runtime}, but PyTorch reports "
            f"{installed_runtime}. Rebuild it with the matching Dockerfile."
        )

    if not torch.cuda.is_available():
        compose_file = (
            "compose.amd.yaml"
            if expected_backend == "rocm"
            else "compose.nvidia.yaml"
        )
        runtime_name = "ROCm/HIP" if expected_backend == "rocm" else "CUDA"
        message = (
            f"{runtime_name} GPU access is not visible inside the container. "
            f"Start it with `docker compose -f {compose_file} up` and verify "
            "that the host GPU driver/runtime is installed."
        )
        if require_gpu:
            raise RuntimeError(message)
        print(f"WARNING: {message} Continuing because REQUIRE_GPU=0.", flush=True)
        return

    # PyTorch intentionally exposes ROCm devices through the torch.cuda API.
    # Executing a real kernel detects driver and architecture mismatches that
    # torch.cuda.is_available() alone can miss on either backend.
    device = torch.device("cuda:0")
    result = (torch.ones(1, device=device) + 1).item()
    torch.cuda.synchronize(device)
    if result != 2:
        raise RuntimeError("GPU smoke test returned an unexpected result.")

    properties = torch.cuda.get_device_properties(device)
    if actual_backend == "rocm":
        runtime = torch.version.hip
        architecture = getattr(properties, "gcnArchName", "unknown")
        backend_label = "ROCm"
    else:
        runtime = torch.version.cuda
        capability = torch.cuda.get_device_capability(device)
        architecture = f"sm_{capability[0]}{capability[1]}"
        backend_label = "CUDA"

    print(
        f"{backend_label} ready: "
        f"torch={torch.__version__}, runtime={runtime}, "
        f"device={properties.name}, architecture={architecture}",
        flush=True,
    )


def main() -> None:
    prepare_directories_and_drop_privileges()
    verify_gpu()

    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if command:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
