from typing import List, Tuple
import os
import pathlib
import typing

import numpy as np
import scipy as sp  # type: ignore
import pandas as pd
from dataclasses import dataclass
from prefect import flow, task, get_run_logger
from typing import Optional

from sklearn.model_selection import KFold  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore


def get_git_root():
    last_cwd = os.getcwd()
    while not os.path.isdir(".git"):
        os.chdir("..")
        cwd = os.getcwd()
        if cwd == last_cwd:
            raise OSError("no .git directory")
        last_cwd = cwd
    return last_cwd


project_root = pathlib.Path(get_git_root())

# Original data from kaggle.
DATA_DIR = project_root / "data" / "original"

# Sparse data dir.
SPARSE_DATA_DIR = project_root / "data" / "sparse"

# Predictions Output data dir.
OUTPUT_DIR = project_root / "data" / "submissions"

# Predictions Output data dir.
REDUCED_DIR = project_root / "data" / "reduced"


for path in [DATA_DIR, SPARSE_DATA_DIR, DATA_DIR, REDUCED_DIR]:
    if not path.is_dir():
        raise ValueError(f"directory not found: f{path}")


@dataclass
class TechnologyRepository:
    name: str

    def __post_init__(self):
        self.train_inputs_path: str = f"{DATA_DIR}/train_{self.name}_inputs.h5"
        self.train_targets_path: str = f"{DATA_DIR}/train_{self.name}_targets.h5"
        self.test_inputs_path: str = f"{DATA_DIR}/test_{self.name}_inputs.h5"
        self.train_inputs_sparse_values_path: str = (
            f"{SPARSE_DATA_DIR}/train_{self.name}_inputs_values.sparse.npz"
        )
        self.train_targets_sparse_values_path: str = (
            f"{SPARSE_DATA_DIR}/train_{self.name}_targets_values.sparse.npz"
        )
        self.test_inputs_sparse_values_path: str = (
            f"{SPARSE_DATA_DIR}/test_{self.name}_inputs_values.sparse.npz"
        )
        self.train_inputs_sparse_idxcol_path: str = (
            f"{SPARSE_DATA_DIR}/train_{self.name}_inputs_idxcol.npz"
        )
        self.train_targets_sparse_idxcol_path: str = (
            f"{SPARSE_DATA_DIR}/train_{self.name}_targets_idxcol.npz"
        )
        self.test_inputs_sparse_idxcol_path: str = (
            f"{SPARSE_DATA_DIR}/test_{self.name}_inputs_idxcol.npz"
        )


multi = TechnologyRepository("multi")
cite = TechnologyRepository("cite")


def format_submission(Y_pred_raw, repo: TechnologyRepository):
    """
    Takes a square matrix of `gene*cell` of the kind usually output
    from models and formats it as necessary for submission to the
    kaggle competition
    """
    logger = get_run_logger()
    logger.info("Loading indices...")
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
    assert len(cell_dict) == len(test_index)
    gene_dict = dict((k, i) for i, k in enumerate(y_columns))
    assert len(gene_dict) == len(y_columns)

    assert len(y_columns) == Y_pred_raw.shape[1]
    assert len(test_index) == Y_pred_raw.shape[0]
    logger.info("Loading evaluation ids...")
    eval_ids = pd.read_parquet(f"{SPARSE_DATA_DIR}/evaluation.parquet")

    # Create two arrays of indices, so that for every row in long `eval_ids`
    # list we have the coordinates of the corresponding value in the
    # model's rectangular output matrix.
    eval_ids_cell_num = eval_ids.cell_id.apply(lambda x: cell_dict.get(x, -1))
    eval_ids_gene_num = eval_ids.gene_id.apply(lambda x: gene_dict.get(x, -1))
    # Eval_id rows that have both and "x" and "y" index are valid
    # TODO: should check that nothing has just one (x or y) index
    valid_multi_rows = (eval_ids_gene_num != -1) & (eval_ids_cell_num != -1)
    # create empty submission series
    submission = pd.Series(
        name="target", index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32
    )
    logger.info("Final step: fill empty submission df...")
    # Neat numpy trick to make a 1d array from 2d based on two arrays:
    # one of "x" coordinates and one of "y" coordinates of the 2d array.
    submission.iloc[valid_multi_rows] = Y_pred_raw[  # type: ignore
        eval_ids_cell_num[valid_multi_rows].to_numpy(),  # type: ignore
        eval_ids_gene_num[valid_multi_rows].to_numpy(),  # type: ignore
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
        raise ValueError(f"Shapes are different. {y_true.shape} != {y_pred.shape}")
    corr_sum = 0
    for i in range(len(y_true)):
        corr_sum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corr_sum / len(y_true)


def test_valid_submission(submission: pd.DataFrame):
    """
    Checks that a submission dataframe is properly formatted for submission
    to Kaggle
    """
    assert submission.index.name == "row_id"
    assert submission.columns == ["target"]
    assert len(submission) == 65744180
    assert submission["target"].isna().sum() == 0


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
class Datasets:
    """
    Holds three basic datasets necessary for an experiment
    """

    train_inputs: typing.Any
    train_targets: typing.Any
    test_inputs: typing.Any


# Prefect functions

def load_data(
    *,
    path: str,
    path_sparse: str,
    max_rows_train: Optional[int] = None,
    sparse: bool = False,
):
    if sparse:
        return sp.sparse.load_npz(path_sparse)[:max_rows_train]
    else:
        return pd.read_hdf(path, start=0, stop=max_rows_train)


@task
def load_train_inputs(*, technology: TechnologyRepository, **kwargs):
    return load_data(
        path=technology.train_inputs_path,
        path_sparse=technology.train_inputs_sparse_values_path,
        **kwargs,
    )


@task
def load_train_targets(*, technology: TechnologyRepository, **kwargs):
    return load_data(
        path=technology.train_targets_path,
        path_sparse=technology.train_targets_sparse_values_path,
        **kwargs,
    )


@task
def load_test_inputs(*, technology: TechnologyRepository, **kwargs):
    return load_data(
        path=technology.test_inputs_path,
        path_sparse=technology.test_inputs_sparse_values_path,
        **kwargs,
    )


@flow
def load_all_data(
    technology: TechnologyRepository,
    max_rows_train: int,
    submit_to_kaggle: bool,
    sparse: bool,
):
    train_inputs = load_test_inputs(
        technology=technology, max_rows_train=max_rows_train, sparse=sparse
    )
    targets_train = load_train_targets(
        technology=technology, max_rows_train=max_rows_train
    )
    if submit_to_kaggle:
        test_inputs = load_test_inputs(technology=technology, sparse=sparse)
    else:
        test_inputs = load_test_inputs(
            technology=technology, max_rows_train=max_rows_train, sparse=sparse
        )
    return Datasets(train_inputs, targets_train, test_inputs)


@task()
def truncated_pca(
    dataset, n_components, return_model: bool = False
) -> Tuple[np.ndarray, Optional[TruncatedSVD]]:
    pca = TruncatedSVD(n_components=n_components)
    # TODO: float16 might be better, saw something in the forum
    pca_features = pca.fit_transform(dataset).astype(np.float32)
    if return_model:
        # TODO: use dataclass here, but wasn't working with prefect
        return pca_features, pca
    else:
        return pca_features, None


@flow
def pca_inputs(
    train_inputs, test_inputs, n_components: int, return_model: bool = False
):
    """
    Stack all input data (including testing inputs) and do PCA on
    everything. Useful since this is not a kernel competition and
    don't need the model to predict on unseen inputs.
    """
    inputs = sp.sparse.vstack([train_inputs, test_inputs])
    assert inputs.shape[0] == train_inputs.shape[0] + test_inputs.shape[0]
    reduced_values, pca_model = truncated_pca(inputs, n_components, return_model)  # type: ignore
    # First len(input_train) rows are input_train
    # Lots of `type: ignore` due to strange typing error from
    # prefect on multiple returns
    pca_train_inputs = reduced_values[: train_inputs.shape[0]]  # type: ignore
    # Last len(input_test) rows are input_test
    pca_test_inputs = reduced_values[train_inputs.shape[0] :]  # type: ignore
    assert (
        pca_train_inputs.shape[0]
        + pca_test_inputs.shape[0]
    # Black doing weird formatting here
    # fmt: off
    ) == reduced_values.shape[0]  # type: ignore
    # fmt: on
    return pca_train_inputs, pca_test_inputs, pca_model  # type: ignore


@task
def fit_and_score_pca_targets(
    train_inputs: np.ndarray,
    pca_train_targets,
    test_inputs: np.ndarray,
    pca_test_targets,
    model,
    pca_model_targets: TruncatedSVD,
) -> Score:
    """
    performs model fit and score where model is predicting a reduced
    pca vector that needs to be converted back to raw data space
    """
    logger = get_run_logger()
    model.fit(train_inputs, pca_train_targets)
    # TODO: review pca de-reduction
    Y_hat = model.predict(test_inputs) @ pca_model_targets.components_
    Y = pca_test_targets @ pca_model_targets.components_
    score = correlation_score(Y, Y_hat)
    logger.info(f"Score: {score}")
    return Score(score=score)


@flow
def k_fold_validation(
    *,
    model,  # model object with `.fit()` and `.predict()` methods
    train_inputs,
    train_targets,
    fit_and_score_func,
    k: int,
    **model_kwargs,
):
    logger = get_run_logger()
    kf = KFold(n_splits=k)
    scores = []
    for fold_index, (train_indices, test_indices) in enumerate(kf.split(train_inputs)):
        fold_train_inputs = train_inputs[train_indices]
        fold_train_targets = train_targets[train_indices]
        fold_test_inputs = train_inputs[test_indices]
        fold_test_targets = train_targets[test_indices]
        logger.info(f"Fitting fold {fold_index}...")
        # Use `.submit` function to make Prefect do tasks concurrently
        score = fit_and_score_func.submit(
            fold_train_inputs,
            fold_train_targets,
            fold_test_inputs,
            fold_test_targets,
            model=model,
            **model_kwargs,
        )
        logger.info(f"Score {score} for fold {fold_index}")
        scores.append(score)
    return scores
