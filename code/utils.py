from genericpath import isdir
from multiprocessing.sharedctypes import Value
import os
import pathlib

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)


def get_git_root():
    last_cwd = os.getcwd()
    while not os.path.isdir('.git'):
        os.chdir('..')
        cwd = os.getcwd()
        if cwd == last_cwd:
            raise OSError('no .git directory')
        last_cwd = cwd
    return last_cwd


# Original data.
DATA_DIR = pathlib.Path(get_git_root()) / "data" / "original"

# Sparse data dir.
SPARSE_DATA_DIR = pathlib.Path(get_git_root()) / "data" / "sparse"

# Output data dir.
OUTPUT_DIR = pathlib.Path(get_git_root()) / "data" / "submissions"

for path in [DATA_DIR, SPARSE_DATA_DIR, DATA_DIR]:
    if not path.is_dir():
        raise ValueError(f"directory not found: f{path}")


@dataclass
class ExperimentRepository:
    name: str

    def __post_init__(self):
        self.train_inputs_path: str = f"{DATA_DIR}/train_{self.name}_inputs.h5"
        self.train_targets_path: str = f"{DATA_DIR}/train_{self.name}_targets.h5"
        self.test_inputs_path: str = f"{DATA_DIR}/test_{self.name}_inputs.h5"
        self.train_inputs_sparse_values_path: str = f"{SPARSE_DATA_DIR}/train_{self.name}_inputs_values.sparse.npz"
        self.train_targets_sparse_values_path: str = f"{SPARSE_DATA_DIR}/train_{self.name}_targets_values.sparse.npz"
        self.test_inputs_sparse_values_path: str = f"{SPARSE_DATA_DIR}/test_{self.name}_inputs_values.sparse.npz"
        self.train_inputs_sparse_idxcol_path: str = f"{SPARSE_DATA_DIR}/train_{self.name}_inputs_idxcol.npz"
        self.train_targets_sparse_idxcol_path: str = f"{SPARSE_DATA_DIR}/train_{self.name}_targets_idxcol.npz"
        self.test_inputs_sparse_idxcol_path: str = f"{SPARSE_DATA_DIR}/test_{self.name}_inputs_idxcol.npz"

# TODO: rename
multi = ExperimentRepository("multi")
cite = ExperimentRepository("cite")


def format_submission(Y_pred_raw, repo: ExperimentRepository):
    """
    Takes a square matrix of `gene*cell` of the kind usually output
    from models and formats it as necessary for submission to the 
    kaggle competition
    """
    logging.info('Loading indices...')
    test_index = np.load(
        repo.test_inputs_sparse_idxcol_path,
        allow_pickle=True,
    )["index"]
    y_columns = np.load(
        repo.train_targets_sparse_idxcol_path,
        allow_pickle=True,
    )["columns"]

    # Maps from row number to cell_id
    cell_dict = dict((k, i) for i, k in enumerate(test_index))
    assert (len(cell_dict) == len(test_index))
    gene_dict = dict((k, i) for i, k in enumerate(y_columns))
    assert (len(gene_dict) == len(y_columns))

    assert (len(y_columns) == Y_pred_raw.shape[1])
    assert (len(test_index) == Y_pred_raw.shape[0])
    logging.info('Loading evaluation ids...')
    eval_ids = pd.read_parquet(f'{SPARSE_DATA_DIR}/evaluation.parquet')

    # Create two arrays of indices, so that for every row in long `eval_ids`
    # list we have the coordinates of the corresponding value in the
    # model's rectangular output matrix.
    eval_ids_cell_num = eval_ids.cell_id.apply(
        lambda x: cell_dict.get(x, -1)
    )
    eval_ids_gene_num = eval_ids.gene_id.apply(
        lambda x: gene_dict.get(x, -1)
    )
    # Eval_id rows that have both and "x" and "y" index are valid
    # TODO: should check that nothing has just one (x or y) index
    valid_multi_rows = (eval_ids_gene_num != -1) & (eval_ids_cell_num != -1)
    # create empty submission series
    submission = pd.Series(
        name='target',
        index=pd.MultiIndex.from_frame(eval_ids),
        dtype=np.float32
    )
    logging.info('Final step: fill empty submission df...')
    # Neat numpy trick to make a 1d array from 2d based on two arrays:
    # one of "x" coordinates and one of "y" coordinates of the 2d array.
    submission.iloc[valid_multi_rows] = Y_pred_raw[    # type: ignore
        eval_ids_cell_num[valid_multi_rows].to_numpy(),  # type: ignore
        eval_ids_gene_num[valid_multi_rows].to_numpy()  # type: ignore
    ]
    return submission


def correlation_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    TODO: write unit test for this
    Scores the predictions according to the competition rules. 
    It is assumed that the predictions are not constant.
    Returns the average of each sample's Pearson correlation coefficient

    take (lightly modified) from: 
    https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart?scriptVersionId=103802624&cellId=7
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes are different. {y_true.shape} != {y_pred.shape}")
    corr_sum = 0
    for i in range(len(y_true)):
        corr_sum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corr_sum / len(y_true)
