#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m remote_go.cli "$@"
