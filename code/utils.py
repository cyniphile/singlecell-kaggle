from typing import List
import os
import pathlib
import typing

import numpy as np
import scipy as sp
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
class TechnologyRepository:
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


multi = TechnologyRepository("multi")
cite = TechnologyRepository("cite")


def format_submission(Y_pred_raw, repo: TechnologyRepository):
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


def test_valid_submission(submission: pd.DataFrame):
    """
    Checks that a submission dataframe is properly formatted for submission
    to Kaggle
    """
    assert submission.index.name == 'row_id'
    assert submission.columns == ['target']
    assert len(submission) == 65744180
    assert submission['target'].isna().sum() == 0


def row_wise_std_scaler(M):
    """ 
    Standard scale values by row. 
    Sklearn StandardScaler has now row-wise option
    """
    std = np.std(M, axis=1).reshape(-1, 1)
    # Make any zero std 1 to avoid numerical problems
    std[std == 0] = 1
    mean = np.mean(M, axis=1).reshape(-1, 1)
    return (M - mean) / std


@dataclass
class Score:
    score: float


@dataclass
class ScoreSummary:
    scores: List[Score]


@dataclass
class ExperimentParameters:
    """
    Base class that holds experiment parameters
    Extensible to hold other parameters as necessary
    """
    # Number of rows to sample from data
    MAX_ROWS_TRAIN: int
    # Whether to create a full submission
    OUTPUT_SUBMISSION: bool
    # The "technology" (cite or multi) to perform the modeling experiment on
    TECHNOLOGY: TechnologyRepository
    NP_RANDOM_SEED = 1000


# Begin Stub classes to extend ExperimentParameters


@dataclass
class PCAInputs:
    INPUTS_PCA_DIMS: int


@dataclass
class PCATargets:
    TARGETS_PCA_DIMS: int


@dataclass
class KFold:
    K_FOLDS: int

# End Stub classes to extend ExperimentParameters


@dataclass
class Datasets:
    """
    Holds three basic datasets necessary for an experiment
    """
    inputs_train: typing.Any
    targets_train: typing.Any
    inputs_test: typing.Any


def load_hdf_data(experiment: ExperimentParameters):
    """
    Load all `.hd5` datasets needed for a given Experiment
    """
    logging.info('Reading `h5d` files...')
    # TODO: extract to method with params  (sparse/dense, technology)
    inputs_train = pd.read_hdf(
        experiment.TECHNOLOGY.train_inputs_path,
        start=0,
        stop=experiment.MAX_ROWS_TRAIN
    )
    targets_train = pd.read_hdf(
        experiment.TECHNOLOGY.train_targets_path,
        start=0,
        stop=experiment.MAX_ROWS_TRAIN
    )
    if experiment.OUTPUT_SUBMISSION:
        inputs_test = pd.read_hdf(
            experiment.TECHNOLOGY.test_inputs_path,
        )
    else:
        inputs_test = pd.read_hdf(
            experiment.TECHNOLOGY.test_inputs_path,
            start=0,
            stop=experiment.MAX_ROWS_TRAIN
        )
    return Datasets(inputs_train, targets_train, inputs_test)


def load_sparse_values_data(experiment: ExperimentParameters):
    """
    Load all `.values.sparse.npz datasets needed for a given Experiment
    Since sklearn algorithms generally don't allow sparse data for targets
    need to continue to just use sparse data there. 
    """
    logging.info('Reading `.sparse.npz` files...')
    inputs_train = sp.sparse.load_npz(
        experiment.TECHNOLOGY.train_inputs_sparse_values_path
    )
    logging.info('Reading `hd5 targets` files...')
    targets_train = pd.read_hdf(
        experiment.TECHNOLOGY.train_targets_path,
        start=0,
        stop=experiment.MAX_ROWS_TRAIN
    )
    if experiment.OUTPUT_SUBMISSION:
        inputs_test = sp.sparse.load_npz(
            experiment.TECHNOLOGY.test_inputs_sparse_values_path
        )
    else:
        inputs_test = pd.read_hdf(
            experiment.TECHNOLOGY.test_inputs_path,
            start=0,
            stop=experiment.MAX_ROWS_TRAIN
        )
    return Datasets(inputs_train, targets_train, inputs_test)
