#!/bin/bash
# Hydrate Copilot tokens for the hermes proxy without putting them in unit files.
set -euo pipefail
export HOME=/home/ubuntu
if token="$(/usr/bin/gh auth token 2>/dev/null)" && [ -n "${token}" ]; then
  export GH_TOKEN="${token}"
  export GITHUB_TOKEN="${token}"
  export COPILOT_GITHUB_TOKEN="${token}"
fi
unset token
exec /home/ubuntu/.local/bin/hermes proxy start --provider copilot --host 127.0.0.1 --port 8648
