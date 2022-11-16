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
import xgboost as xgb


# TODO: refactor to not be a decorator, instead just run as a subflow in the
# `full_submission` branch.
@run_or_get_cache
@flow(
    name="XGBoost with Input and Target PCA",
    description="XGBoost with input and output PCA",
    # TODO: could be a way to turn this into a cache result
    # persist_result=True,
    # result_storage=LocalFileSystem(basepath=str(utils.OUTPUT_DIR)),
)

# TODO: add params
# max_depth: The maximum depth per tree. A deeper tree might increase the performance, but also the complexity and chances to overfit.
# The value must be an integer greater than 0. Default is 6.
# learning_rate: The learning rate determines the step size at each iteration while your model optimizes toward its objective. A low learning rate makes computation slower, and requires more rounds to achieve the same reduction in residual error as a model with a high learning rate. But it optimizes the chances to reach the best optimum.
# The value must be between 0 and 1. Default is 0.3.
# n_estimators: The number of trees in our ensemble. Equivalent to the number of boosting rounds.
# The value must be an integer greater than 0. Default is 100.
# NB: In the standard library, this is referred as num_boost_round.
# colsample_bytree: Represents the fraction of columns to be randomly sampled for each tree. It might improve overfitting.
# The value must be between 0 and 1. Default is 1.
# subsample: Represents the fraction of observations to be sampled for each tree. A lower values prevent overfitting but might lead to under-fitting.
# The value must be between 0 and 1. Default is 1.
# Regularization parameters:

# alpha (reg_alpha): L1 regularization on the weights (Lasso Regression). When working with a large number of features, it might improve speed performances. It can be any integer. Default is 0.
# lambda (reg_lambda): L2 regularization on the weights (Ridge Regression). It might help to reduce overfitting. It can be any integer. Default is 1.
# gamma: Gamma is a pseudo-regularisation parameter (Lagrangian multiplier), and depends on the other parameters. The higher Gamma is, the higher the regularization. It can be any integer. Default is 0.


def xgb_flow(
    # default params used for testing
    max_rows_train: int = 1_000,
    full_submission: bool = False,
    technology: TechnologyRepository = utils.cite,
    inputs_pca_dims: int = 2,
    targets_pca_dims: int = 2,
    k_folds: int = 2,
    sparse: bool = True,
    custom_loss=True,
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
        if custom_loss:
            model = xgb.XGBRegressor(objective=utils.correlation_loss_grad)
        else:
            model = xgb.XGBRegressor()

        if full_submission:
            mlflow.sklearn.autolog()
            Y_hat = fit_and_score_pca_input_output(
                train_inputs=data.train_inputs.values,
                train_targets=data.train_targets.values.values,  # type: ignore
                test_inputs=data.test_inputs.values,  # type: ignore
                model=model,
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
                model=model,
                train_inputs=data.train_inputs.values,
                train_targets=data.train_targets.values.values,  # type: ignore
                fit_and_score_task=utils.fit_and_score_pca_input_output,
                inputs_pca_dims=inputs_pca_dims,
                targets_pca_dims=targets_pca_dims,
            )
            return ScoreSummary(scores)


if __name__ == "__main__":
    xgb_flow(max_rows_train=10000)  # type: ignore
