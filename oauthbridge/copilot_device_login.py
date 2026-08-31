#!/usr/bin/env python3
"""Start GitHub Copilot device-code login and persist the token into Hermes."""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/.hermes/hermes-agent")
from hermes_cli.copilot_auth import COPILOT_OAUTH_CLIENT_ID  # type: ignore

STATE = Path("/home/ubuntu/vault/projects/llm-router-dash/copilot-oauth-state.json")
RESULT = Path("/home/ubuntu/vault/projects/llm-router-dash/copilot-oauth-result.txt")


def main() -> int:
    device_code_url = "https://github.com/login/device/code"
    access_token_url = "https://github.com/login/oauth/access_token"
    data = urllib.parse.urlencode(
        {"client_id": COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"}
    ).encode()
    req = urllib.request.Request(
        device_code_url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "HermesAgent/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        device_data = json.loads(resp.read().decode())
    verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(int(device_data.get("interval", 5)), 1)
    if not device_code or not user_code:
        print("github did not return a device code")
        return 1
    STATE.write_text(json.dumps({"verification_uri": verification_uri, "user_code": user_code}))
    print(f"URL {verification_uri}")
    print(f"CODE {user_code}")
    deadline = time.monotonic() + 240
    token = None
    while time.monotonic() < deadline:
        time.sleep(interval + 1)
        poll = urllib.parse.urlencode(
            {
                "client_id": COPILOT_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }
        ).encode()
        preq = urllib.request.Request(
            access_token_url,
            data=poll,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "HermesAgent/1.0",
            },
        )
        try:
            with urllib.request.urlopen(preq, timeout=15) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:
            print(f"poll_err {exc}")
            continue
        if payload.get("access_token"):
            token = payload["access_token"]
            break
        err = payload.get("error")
        if err and err not in {"authorization_pending", "slow_down"}:
            print(f"oauth_error {err} {payload.get('error_description')}")
            return 2
        print("pending")
    if not token:
        print("timeout")
        return 3
    # persist via hermes pool without printing the token
    from agent.credential_pool import load_pool

    pool = load_pool("copilot")
    # store as oauth-ish access token via hermes auth add api-key using env
    import os
    import subprocess

    env = os.environ.copy()
    # don't log
    proc = subprocess.run(
        [
            "/home/ubuntu/.local/bin/hermes",
            "auth",
            "add",
            "copilot",
            "--type",
            "api-key",
            "--label",
            "copilot-oauth-device",
            "--api-key",
            token,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    RESULT.write_text(f"ok exit={proc.returncode} stdout={proc.stdout[-200:]} stderr={proc.stderr[-200:]}\n")
    print(f"saved_exit {proc.returncode} token_len {len(token)}")
    return 0 if proc.returncode == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
