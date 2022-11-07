#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
poetry export -f requirements.txt --output requirements.txt 
pip-compile --generate-hashes -o requirements.txt requirements.txt    