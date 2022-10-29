import utils
from typing import List
import inspect
import argparse
from utils import (
    TechnologyRepository,
    fit_and_score_pca_targets,
    k_fold_validation,
    truncated_pca,
    run_or_get_cache,
    pca_inputs,
    load_all_data,
    ScoreSummary,
    Datasets,
)
import numpy as np
from prefect import flow, get_run_logger
from prefect.filesystems import LocalFileSystem
import mlflow  # type: ignore
from sklearn.gaussian_process.kernels import RBF  # type: ignore
from sklearn.kernel_ridge import KernelRidge  # type: ignore

# from prefect_dask.task_runners import DaskTaskRunner  # type: ignore

# By default, Prefect makes a best effort to compute a
# table hash of the .py file in which the flow is defined to
# automatically detect when your code changes.
@run_or_get_cache
@flow(
    name="RBF with Input and Target PCA",
    description="Based on last year's winner of RNA->Prot",
    # persist_result=True,
    # result_storage=LocalFileSystem(basepath=str(utils.OUTPUT_DIR)),
    # task_runner=DaskTaskRunner(),
)
def last_year_rbf_flow(
    # default params used for testing
    max_rows_train: int = 1_000,
    full_submission: bool = False,
    technology: TechnologyRepository = utils.cite,
    inputs_pca_dims: int = 2,
    targets_pca_dims: int = 2,
    k_folds: int = 2,
    scale: float = 10,  # RBF scale param. Higher means more model complexity
    alpha: float = 0.2,  # Regularization param. More is more regularization.
):
    with mlflow.start_run():
        # log all inputs into mlflow
        mlflow.log_params(locals())
        # log name of function so it can be called later
        mlflow.log_param("flow_function", inspect.currentframe().f_code.co_name)  # type: ignore
        logging = get_run_logger()
        data: Datasets = load_all_data(  # type: ignore
            technology=technology,
            max_rows_train=max_rows_train,
            full_submission=full_submission,
            sparse=True,
        )
        train_inputs, train_targets, test_inputs = (
            data.train_inputs,
            data.train_targets,
            data.test_inputs,
        )
        pca_train_inputs, pca_test_inputs, _ = pca_inputs(  # type: ignore
            train_inputs, test_inputs, inputs_pca_dims
        )
        pca_train_targets, pca_model_targets = truncated_pca.submit(  # type: ignore
            train_targets,
            targets_pca_dims,
            return_model=True,
        ).result()
        # TODO: ensure float32 is best
        train_norm = utils.row_wise_std_scaler(pca_train_inputs).astype(np.float32)  # type: ignore
        kernel = RBF(length_scale=scale)
        krr = KernelRidge(alpha=alpha, kernel=kernel)  # type: ignore
        if full_submission:
            mlflow.sklearn.autolog()
            test_norm = utils.row_wise_std_scaler(pca_test_inputs).astype(np.float32)  # type: ignore
            logging.info("Fit full model on all training data")
            krr.fit(train_norm, pca_train_targets)  # type: ignore
            logging.info("Predict on full submission inputs")
            Y_hat = krr.predict(test_norm) @ pca_model_targets.components_  # type: ignore
            formatted_submission = utils.format_submission.submit(
                Y_hat, technology
            ).result()
            return formatted_submission
        else:
            mlflow.sklearn.autolog()
            scores: List[utils.Score] = k_fold_validation(
                model=krr,
                train_inputs=train_norm,
                train_targets=pca_train_targets,  # type: ignore
                fit_and_score_task=fit_and_score_pca_targets,
                k=k_folds,
                pca_model_targets=pca_model_targets,  # type: ignore
            )
            score_summary = ScoreSummary(scores)


if __name__ == "__main__":
    # Run script as a test
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_submission", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Should be a 1:1 mapping of args to function params
    args_dict = vars(args)
    if args.full_submission:
        last_year_rbf_flow(**args_dict)  # type: ignore
    else:
        # run with default test params, and caching disabled
        last_year_rbf_flow(ignore_cache=True, skip_caching=True, technology=utils.multi)  # type: ignore
        last_year_rbf_flow(ignore_cache=True, skip_caching=True)  # type: ignore
