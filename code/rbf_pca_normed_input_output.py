import utils
import argparse
from prefect_dask.task_runners import DaskTaskRunner

# from prefect.client.schemas import State
from utils import (
    fit_and_score_pca_targets,
    k_fold_validation,
    truncated_pca,
    pca_inputs,
    load_all_data,
)
import numpy as np
import pandas as pd
from utils import Datasets
from prefect import flow, get_run_logger

from sklearn.gaussian_process.kernels import RBF  # type: ignore
from sklearn.kernel_ridge import KernelRidge  # type: ignore

# By default, Prefect makes a best effort to compute a
# table hash of the .py file in which the flow is defined to
# automatically detect when your code changes.
@flow(
    name="RBF with Input and Target PCA",
    description="Based on last year's winner of RNA->Prot",
)
def last_year_rbf_flow(
    max_rows_train=1000,
    submit_to_kaggle=False,
    technology=utils.cite,
    inputs_pca_dims=4,
    targets_pca_dims=4,
    k_folds=2,
    scale=10,  # RBF scale param. Higher means more model complexity
    alpha=0.2,  # Regularization param. More is more regularization.
):
    logger = get_run_logger()
    if technology == utils.multi:
        data: Datasets = load_all_data(
            technology=technology,
            max_rows_train=max_rows_train,
            submit_to_kaggle=submit_to_kaggle,
            sparse=True,
        )
    else:
        data: Datasets = load_all_data(
            technology=technology,
            max_rows_train=max_rows_train,
            submit_to_kaggle=submit_to_kaggle,
            sparse=True,
        )
    train_inputs, targets_train, test_inputs = (
        data.train_inputs,
        data.train_targets,
        data.test_inputs,
    )
    pca_train_inputs, pca_test_inputs, _ = pca_inputs(
        train_inputs, test_inputs, inputs_pca_dims
    )
    pca_targets_train, pca_model_targets = truncated_pca(
        targets_train, targets_pca_dims, return_model=True
    )
    train_norm = utils.row_wise_std_scaler(pca_train_inputs).astype(np.float32)
    del pca_train_inputs
    kernel = RBF(length_scale=scale)
    krr = KernelRidge(alpha=alpha, kernel=kernel)  # type: ignore
    scores = k_fold_validation(
        model=krr,
        train_inputs=train_norm,
        train_targets=pca_targets_train,
        fit_and_score_func=fit_and_score_pca_targets,
        k=k_folds,
        pca_model_targets=pca_model_targets,
    )
    logger.info(f"K-Fold complete. Scores: {scores}")

    test_norm = utils.row_wise_std_scaler(pca_test_inputs).astype(np.float32)
    del pca_test_inputs
    if submit_to_kaggle:
        # TODO: extract to utils method
        OTHER_FILENAME = "cite_rbf_with_multi_linear"
        OTHER_SUBMISSION_PATH = utils.OUTPUT_DIR / f"{OTHER_FILENAME}.csv"
        # fit model on downsampled data
        krr.fit(train_norm, pca_targets_train)
        # predict on full submission inputs
        Y_hat = krr.predict(test_norm) @ pca_model_targets.components_  # type: ignore
        # Format this experiment for submission
        this_submission = utils.format_submission(Y_hat, technology)
        # Load other submission which includes predictions
        # for alternate tech
        other_submission = pd.read_csv(OTHER_SUBMISSION_PATH, index_col=0)
        # drop multi-index to align with other submission
        reindexed_submission_this = pd.DataFrame(this_submission.reset_index(drop=True))
        # Merge with separate predictions for other technology
        merged = reindexed_submission_this["target"].fillna(
            other_submission[reindexed_submission_this["target"].isna()]["target"]
        )
        # put into dataframe with proper column names
        formatted_submission = pd.DataFrame(merged, columns=["target"])
        formatted_submission.index.name = "row_id"
        utils.test_valid_submission(formatted_submission)
        # write full predictions to csv
        logger.info(
            utils.OUTPUT_DIR / f"{technology.name}_rbf_with_{OTHER_FILENAME}.csv"
        )
        # TODO: change to prefect `LocalResult`
        formatted_submission.to_csv(
            utils.OUTPUT_DIR / f"{technology.name}_rbf_with_{OTHER_FILENAME}.csv"
        )
    else:
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_job", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.full_job:
        last_year_rbf_flow(
            max_rows_train=1_000,
            submit_to_kaggle=False,
            technology=utils.multi,
            inputs_pca_dims=5,
            targets_pca_dims=4,
            k_folds=3,
            scale=10,  # RBF scale param. Higher means more model complexity
            alpha=0.2,  # Regularization param. More is more regularization.
        )
    else:
        scores = last_year_rbf_flow(  # type: ignore
            max_rows_train=1_000,
        )
        # Need `.result()` as results are now asynchronous with dask
        assert sum([s.result().score for s in scores]) / len(scores) > 0.9  # type: ignore
