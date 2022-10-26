#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cd code

python ./rbf_pca_normed_input_output.py --full_submission