"""Fail-fast CUDA validation for the Docker entry point."""

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

    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)

    if os.geteuid() == 0:
        raise RuntimeError("Failed to drop root privileges.")


def verify_cuda() -> None:
    require_cuda = env_flag("REQUIRE_CUDA", True)

    if not torch.cuda.is_available():
        message = (
            "CUDA is not visible inside the container. Start it with "
            "`docker compose up` or `docker run --gpus all ...`."
        )
        if require_cuda:
            raise RuntimeError(message)
        print(f"WARNING: {message} Continuing because REQUIRE_CUDA=0.", flush=True)
        return

    # Executing a real kernel detects runtime/driver/architecture mismatches that
    # torch.cuda.is_available() alone can miss.
    device = torch.device("cuda:0")
    result = (torch.ones(1, device=device) + 1).item()
    torch.cuda.synchronize(device)
    if result != 2:
        raise RuntimeError("CUDA smoke test returned an unexpected result.")

    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    print(
        "CUDA ready: "
        f"torch={torch.__version__}, runtime={torch.version.cuda}, "
        f"device={properties.name}, capability=sm_{capability[0]}{capability[1]}",
        flush=True,
    )


def main() -> None:
    prepare_directories_and_drop_privileges()
    verify_cuda()

    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if command:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
