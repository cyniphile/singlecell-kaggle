#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cd code
# Will not work on windows as dumps temporary notebook to /dev/null	
# See https://github.com/nteract/papermill/issues/405
papermill last_year_rbf_prefect.ipynb /dev/null -p IS_TEST True --log-output