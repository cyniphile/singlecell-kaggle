#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt