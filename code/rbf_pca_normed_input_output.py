import utils
import os
from typing import List, Optional
import inspect
from utils import (
    fit_and_score_pca_input_output,
    TechnologyRepository,
    k_fold_validation,
    run_or_get_cache,
    load_all_data,
    ScoreSummary,
    Datasets,
)
from prefect import flow
import mlflow  # type: ignore
from sklearn.gaussian_process.kernels import RBF  # type: ignore
from sklearn.kernel_ridge import KernelRidge  # type: ignore


# TODO: refactor to not be a decorator, instead just run as a subflow in the
# `full_submission` branch.
@run_or_get_cache
@flow(
    name="RBF with Input and Target PCA",
    description="Based on last year's winner of RNA->Prot",
    # TODO: could be a way to turn this into a cache result
    # persist_result=True,
    # result_storage=LocalFileSystem(basepath=str(utils.OUTPUT_DIR)),
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
    sparse: bool = True,
):
    with mlflow.start_run():
        # log all inputs into mlflow
        mlflow.log_params(locals())
        # log name of function so it can be called later
        mlflow.log_param("flow_function", inspect.currentframe().f_code.co_name)  # type: ignore
        mlflow.log_param("flow_filepath", os.path.realpath(__file__))
        data: Datasets = load_all_data(  # type: ignore
            technology=technology,
            max_rows_train=max_rows_train,
            full_submission=full_submission,
            sparse=sparse,
        )
        kernel = RBF(length_scale=scale)
        krr = KernelRidge(alpha=alpha, kernel=kernel)  # type: ignore

        if full_submission:
            mlflow.sklearn.autolog()
            Y_hat = fit_and_score_pca_input_output(
                train_inputs=data.train_inputs.values,
                train_targets=data.train_targets.values.values,  # type: ignore
                test_inputs=data.test_inputs.values,  # type: ignore
                model=krr,
                inputs_pca_dims=inputs_pca_dims,
                targets_pca_dims=targets_pca_dims,
            )
            formatted_submission = utils.format_submission.submit(
                Y_hat, technology
            ).result()
            return formatted_submission
        else:
            mlflow.sklearn.autolog()
            scores: List[utils.Score] = k_fold_validation(
                k=k_folds,
                model=krr,
                train_inputs=data.train_inputs.values,
                train_targets=data.train_targets.values.values,  # type: ignore
                fit_and_score_task=utils.fit_and_score_pca_input_output,
                inputs_pca_dims=inputs_pca_dims,
                targets_pca_dims=targets_pca_dims,
            )
            return ScoreSummary(scores)


if __name__ == "__main__":
    # last_year_rbf_flow(max_rows_train=40_000)  # type: ignore
    # last_year_rbf_flow(max_rows_train=40_000, technology=utils.multi)  # type: ignore
    last_year_rbf_flow(max_rows_train=None)  # type: ignore
