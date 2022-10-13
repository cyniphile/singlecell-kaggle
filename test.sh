#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
cd code
# Will not work on windows as dumps temporary notebook to /dev/null	
# See https://github.com/nteract/papermill/issues/405
papermill  Last\ Year\ RNA-\>Prot\ Test.ipynb /dev/null -p IS_TEST True --log-output
