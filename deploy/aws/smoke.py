"""Run a disposable, network-isolated container smoke test; never push an image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="scone-aws:local")
    image = parser.parse_args().image
    if not isinstance(image, str) or not image or image.startswith("-"):
        parser.error("image must be a nonempty image reference")
    env = {key: value for key in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
                                 "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
           if (value := os.environ.get(key)) is not None}
    env["SCONE_API_KEY"] = secrets.token_hex(32)
    container = subprocess.check_output(
        ["docker", "run", "--detach", "--pull=never", "--network=none", "--read-only",
         "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128",
         "--memory=512m", "--tmpfs=/tmp:rw,noexec,nosuid,size=32m",
         "--tmpfs=/data:rw,nosuid,uid=10001,gid=10001,size=64m",
         "--env=SCONE_API_KEY", "--env=AWS_EC2_METADATA_DISABLED=true", image],
        text=True, env=env, timeout=30,
    ).strip()
    try:
        probe = Path(__file__).with_name("smoke_probe.py").read_text()
        subprocess.run(["docker", "exec", "--interactive", container, "python", "-"],
                       input=probe, text=True, check=True, timeout=45, env=env)
        subprocess.run(["docker", "exec", container, "python", "/opt/healthcheck.py"],
                       check=True, timeout=10, env=env)
    finally:
        subprocess.run(["docker", "rm", "--force", container], check=True,
                       stdout=subprocess.DEVNULL, timeout=30, env=env)
    print("PASS: Python 3.14, AWS/Qdrant imports, nonroot, isolated network, health, API-only routes, auth, synthetic SQLite write and recall")


if __name__ == "__main__":
    main()
