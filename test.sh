#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cd code
# papermill last_year_rbf_prefect.ipynb ../code/tmp/test_output.ipynb -p IS_TEST True --log-output
python ./rbf_pca_normed_input_output.py