from typing import List, Tuple, Union
import jax
from jax import numpy as jnp, jit
import functools
import sys
import dataclasses, json
import importlib.util
import logging
import hashlib
import inspect
import json
import os
import pathlib
import mlflow  # type: ignore
import numpy as np
import scipy as sp  # type: ignore
import pandas as pd
from dataclasses import dataclass
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from typing import Optional
from sklearn.model_selection import KFold  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore

logging.getLogger().setLevel(logging.INFO)


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
try:
    assert os.environ["SATURN_RESOURCE_NAME"] == "openproblems-bio-2022"
    DATA_DIR = project_root / ".." / "input" / "open-problems-multimodal"
except KeyError:
    DATA_DIR = project_root / "data" / "original"

# Sparse data dir.
SPARSE_DATA_DIR = project_root / "data" / "sparse"

# Predictions Output data dir.
OUTPUT_DIR = project_root / "data" / "submissions"

METADATA_PATH = SPARSE_DATA_DIR / "metadata.parquet"


for path in [DATA_DIR, SPARSE_DATA_DIR, DATA_DIR]:
    if not path.is_dir():
        raise ValueError(f"directory not found: {path}")

SPLIT_TYPES = ["train", "test"]
DATA_TYPES = ["inputs", "targets"]


@dataclass
class TechnologyRepository:
    name: str

    def __post_init__(self):
        for split in SPLIT_TYPES:
            for data_type in DATA_TYPES:
                _full: str = f"{DATA_DIR}/{split}_{self.name}_{data_type}.h5"
                _sparse: str = f"{SPARSE_DATA_DIR}/{split}_{self.name}_{data_type}_values.sparse.npz"
                _sparse_idx: str = (
                    f"{SPARSE_DATA_DIR}/{split}_{self.name}_{data_type}_idxcol.npz"
                )
                setattr(self, f"{split}_{data_type}_path", _full)
                setattr(self, f"{split}_{data_type}_sparse_values_path", _sparse)
                setattr(self, f"{split}_{data_type}_sparse_idxcol_path", _sparse_idx)


multi = TechnologyRepository("multi")
cite = TechnologyRepository("cite")


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


def test_valid_submission(submission: pd.DataFrame) -> None:
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
class DensePathDescription:
    path: str


@dataclass
class SparsePathDescription(DensePathDescription):
    idx_path: str


PathDescription = Union[DensePathDescription, SparsePathDescription]


@dataclass
class Score:
    score: float


@dataclass
class ScoreSummary:
    scores: List[Score]


@dataclass
class DenseDataset:
    values: pd.DataFrame


@dataclass
class SparseDataset(DenseDataset):
    index: np.lib.npyio.NpzFile  # type: ignore


Dataset = Union[SparseDataset, DenseDataset]


@dataclass
class Datasets:
    """
    Holds three basic datasets necessary for an experiment
    """

    # _: dataclasses.KW_ONLY
    train_inputs: Dataset
    train_targets: Dataset
    test_inputs: Dataset


# Prefect pipeline functions


def load_data(
    *,
    path_description: PathDescription,
    max_rows: int = 0,
) -> Dataset:  # type: ignore
    path = path_description.path
    if max_rows == 0:
        if type(path_description) == SparsePathDescription:
            idx_path = path_description.idx_path  # type: ignore
            sparse_data = sp.sparse.load_npz(path)  # type: ignore
            sparse_index = np.load(idx_path, allow_pickle=True)
            return SparseDataset(values=sparse_data, index=sparse_index)
        if type(path_description) == DensePathDescription:
            df: pd.DataFrame = pd.read_hdf(path)  # type: ignore
            return DenseDataset(values=df)  # type: ignore
        else:
            raise RuntimeError()
    else:
        # make np return same random string each time
        np.random.seed(0)
        if type(path_description) == SparsePathDescription:
            idx_path = path_description.idx_path  # type: ignore
            sparse_data = sp.sparse.load_npz(path)  # type: ignore
            random_indices = np.random.uniform(
                0,
                sparse_data.shape[0],
                min(max_rows, sparse_data.shape[0]),
            )
            filtered_sparse_data = sparse_data[random_indices]
            sparse_index = np.load(idx_path, allow_pickle=True)
            return SparseDataset(values=filtered_sparse_data, index=sparse_index)
        elif type(path_description) == DensePathDescription:
            df: pd.DataFrame = pd.read_hdf(path)  # type: ignore
            random_indices = np.random.uniform(0, df.shape[0], max_rows)  # type: ignore
            random_subset = df.iloc[random_indices]  # type: ignore
            return DenseDataset(values=random_subset)  # type: ignore
        else:
            raise RuntimeError()


class Loaders:
    pass


def make_loader(split_type, data_type):
    @task(name=f"load_{split_type}_{data_type}")
    def _loader(
        technology: TechnologyRepository,
        sparse: bool = False,
        max_rows: int = 0,
    ):
        if sparse:
            idx_path = getattr(
                technology, f"{split_type}_{data_type}_sparse_idxcol_path"
            )
            values_path = getattr(
                technology, f"{split_type}_{data_type}_sparse_values_path"
            )
            path_description = SparsePathDescription(values_path, idx_path)
            return load_data(
                path_description=path_description,
                max_rows=max_rows,
            )
        else:
            values_path = getattr(technology, f"{split_type}_{data_type}_path")
            return load_data(
                path_description=DensePathDescription(values_path),
                max_rows=max_rows,
            )

    return _loader


for split in SPLIT_TYPES:
    for data in DATA_TYPES:
        _loader = make_loader(split, data)
        func_name = f"load_{split}_{data}"
        setattr(Loaders, func_name, _loader)


@flow
def load_all_data(
    technology: TechnologyRepository,
    max_rows_train: int = 0,
    full_submission: bool = False,
    sparse: bool = False,
):
    train_inputs = Loaders.load_train_inputs.submit(  # type: ignore
        technology=technology,
        max_rows=max_rows_train,
        sparse=sparse,
    ).result()
    # Targets need to be in dense format for sklearn training :-(
    train_targets = Loaders.load_train_targets.submit(  # type: ignore
        technology=technology,
        max_rows=max_rows_train,
    ).result()
    # If submitting to kaggle need to load full test_inputs to generate
    # a complete and valid submission
    if full_submission:
        test_inputs = Loaders.load_test_inputs.submit(  # type: ignore
            technology=technology, sparse=sparse
        ).result()
    else:
        test_inputs = Loaders.load_test_inputs.submit(  # type: ignore
            technology=technology,
            max_rows=max_rows_train,
            sparse=sparse,
        ).result()
    return Datasets(
        train_inputs=train_inputs, train_targets=train_targets, test_inputs=test_inputs
    )


@task
def load_metadata():
    return pd.read_parquet(METADATA_PATH).set_index("cell_id")


# @flow
# TODO: don't match for saturn cloud
# def merge_with_metadata(*, train_inputs, test_inputs):
#     metadata = load_metadata()
#     match train_inputs, test_inputs:
#         case (
#             SparseDataset(values=train_inputs_values, index=train_inputs_index),
#             SparseDataset(values=test_inputs_values, index=test_inputs_index),
#         ):
#             pass
#         case (DenseDataset(train_inputs), DenseDataset(test_inputs)):
#             merged_train = pd.merge(
#                 train_inputs, metadata, left_index=True, right_index=True
#             )
#             assert merged_train.shape[0] == train_inputs.shape[0]
#             merged_test = pd.merge(
#                 test_inputs, metadata, left_index=True, right_index=True
#             )
#             assert merged_test.shape[0] == test_inputs.shape[0]
#             return merged_train, merged_test


@task(cache_key_fn=task_input_hash)
def truncated_pca(dataset, n_components) -> Tuple[np.ndarray, Optional[TruncatedSVD]]:
    pca = TruncatedSVD(n_components=n_components)
    # TODO: float16 might be better, saw something in the forum
    pca_features = pca.fit_transform(dataset).astype(np.float32)
    # TODO: use dataclass here, but wasn't working with prefect
    return pca_features, pca


@flow
def pca_inputs(train_inputs, test_inputs, n_components: int):
    """
    Stack all input data (including testing inputs) and do PCA on
    everything. Useful since this is not a kernel competition and
    don't need the model to predict on unseen inputs.
    """
    inputs = sp.sparse.vstack([train_inputs, test_inputs])
    assert inputs.shape[0] == train_inputs.shape[0] + test_inputs.shape[0]
    reduced_values, pca_model = truncated_pca.submit(  # type: ignore
        inputs, n_components
    ).result()
    # First len(input_train) rows are input_train
    # Lots of `type: ignore` due to strange typing error from
    # prefect on multiple returns
    # TODO: ensure stacking is preserved, not mixing train test somehow?
    pca_train_inputs = reduced_values[: train_inputs.shape[0]]  # type: ignore
    # Last len(input_test) rows are input_test
    pca_test_inputs = reduced_values[train_inputs.shape[0] :]  # type: ignore
    # fmt: off
    assert (
        pca_train_inputs.shape[0] + pca_test_inputs.shape[0]
    ) == reduced_values.shape[0]  # type: ignore
    # fmt: on
    assert pca_train_inputs.shape[0] == train_inputs.shape[0]
    assert pca_test_inputs.shape[0] == test_inputs.shape[0]

    return pca_train_inputs, pca_test_inputs, pca_model  # type: ignore


@flow
def k_fold_validation(
    *,
    model,  # model object with `.fit()` and `.predict()` methods
    train_inputs,
    train_targets,
    fit_and_score_task,
    k: int,
    **model_kwargs,
):
    logging = get_run_logger()
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    scores = []
    for fold_index, (train_indices, test_indices) in enumerate(kf.split(train_inputs)):
        fold_train_inputs = train_inputs[train_indices]
        fold_train_targets = train_targets[train_indices]
        fold_test_inputs = train_inputs[test_indices]
        fold_test_targets = train_targets[test_indices]
        logging.info(f"Fitting fold {fold_index}...")
        score = fit_and_score_task(
            train_inputs=fold_train_inputs,
            train_targets=fold_train_targets,
            test_inputs=fold_test_inputs,
            test_targets=fold_test_targets,
            model=model,
            **model_kwargs,
        )
        logging.info(f"Score {score} for fold {fold_index}")
        scores.append(score)
    return scores


@task
def format_submission(Y_pred_raw, technology: TechnologyRepository):
    # TODO: speed up with jax
    """
    Takes a square matrix of `gene*cell` of the kind usually output
    from models and formats it as necessary for submission to the
    kaggle competition
    """
    logging = get_run_logger()
    logging.info("Loading indices...")
    test_index = np.load(
        technology.test_inputs_sparse_idxcol_path,  # type: ignore
        allow_pickle=True,
    )["index"]
    y_columns = np.load(
        technology.train_targets_sparse_idxcol_path,  # type: ignore
        allow_pickle=True,
    )["columns"]

    # Maps from row number to cell_id
    cell_dict = dict((k, i) for i, k in enumerate(test_index))
    assert len(cell_dict) == len(test_index)
    gene_dict = dict((k, i) for i, k in enumerate(y_columns))
    assert len(gene_dict) == len(y_columns)

    assert len(y_columns) == Y_pred_raw.shape[1]
    assert len(test_index) == Y_pred_raw.shape[0]
    logging.info("Loading evaluation ids...")
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
    logging.info("Final step: fill empty submission df...")

    # Neat numpy trick to make a 1d array from 2d based on two arrays:
    # one of "x" coordinates and one of "y" coordinates of the 2d array.
    submission.iloc[valid_multi_rows] = Y_pred_raw[  # type: ignore
        eval_ids_cell_num[valid_multi_rows].to_numpy(),  # type: ignore
        eval_ids_gene_num[valid_multi_rows].to_numpy(),  # type: ignore
    ]
    # Need to convert to dataframe else prefect slows things down a lot
    # https://github.com/PrefectHQ/prefect/issues/7065#issuecomment-1292345882
    return pd.DataFrame(submission)


# @task # very slow as task
def _merge_submission(
    this_technology_predictions,
    other_technology_predictions,
):
    """
    merge predictions for two technologies to make a full submission for both
    """
    # drop multi-index to align with other submission
    reindexed_submission_this = this_technology_predictions.reset_index(drop=True)
    # Merge with separate predictions for other technology
    merged = reindexed_submission_this["target"].fillna(
        other_technology_predictions[reindexed_submission_this["target"].isna()][
            "target"
        ]
    )
    # put into dataframe with proper column names
    formatted_submission = pd.DataFrame(merged, columns=["target"])
    formatted_submission.index.name = "row_id"
    test_valid_submission(formatted_submission)
    logging.info("finished merging")
    return formatted_submission


# @task
def _submit_to_kaggle(merged_submission, submission_message: str):
    # write full predictions to csv
    file_safe_message = "".join(x for x in submission_message if x.isalnum())
    submission_file_name = f"{str(OUTPUT_DIR)}/{file_safe_message}.csv"
    logging.info(f"Writing {submission_file_name} locally for submission")
    merged_submission.to_csv(submission_file_name)
    logging.info(f"Submitting {submission_file_name} to kaggle ")
    os.system(
        f'kaggle competitions submit -c open-problems-multimodal -f {submission_file_name} -m "{submission_message}"'
    )


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    taken from https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses
    used to encode dataclasses into json
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class SubmissionExperiments:
    cite_mlflow_id: str
    multi_mlflow_id: str

    def __str__(self):
        return f"cite-{self.cite_mlflow_id}--multi-{self.multi_mlflow_id}"


# @flow  # again, was slow on large inputs
def _merge_and_submit(df_cite, df_multi, experiment_ids: SubmissionExperiments):
    logging.info("starting merge")
    merged = _merge_submission(df_cite, df_multi)
    _submit_to_kaggle(merged, str(experiment_ids))


@flow
def run_or_get_cache(flow):
    """
    TODO: use https://docs.prefect.io/concepts/states/#states to mark if read.
    Decorator for flows. Determine if it has been run before. If not
    run and save results in arrow/feather file; if so return saved results.
    """

    # TODO: not working if flow was run with "full_submission=True" before
    logging = get_run_logger()

    @functools.wraps(flow)
    def wrapper(**kwargs):
        try:
            ignore_cache = kwargs.pop("ignore_cache")
        except KeyError:
            ignore_cache = False
        try:
            skip_caching = kwargs.pop("skip_caching")
        except KeyError:
            skip_caching = False
        default_args_str = str(inspect.signature(flow).parameters)
        overridden_args_str = json.dumps(
            kwargs, sort_keys=True, cls=EnhancedJSONEncoder
        )
        hash_base = (default_args_str + overridden_args_str).encode("utf-8")
        args_hash = str(int(hashlib.md5(hash_base).hexdigest(), 16))
        filename = "-".join([flow.__name__, args_hash]) + ".arrow"
        file_path = f"{OUTPUT_DIR}/{filename}"
        if os.path.exists(file_path) and not ignore_cache:
            logging.info(f"found cache {file_path}, skipping recalculation...")
            submission = pd.read_feather(file_path)
            return submission
        else:
            logging.info(f"no cache found, running flow {flow.__name__}")
            # If submission returns a ScoreSummary don't try to cache.
            # Also don't cache if we say not to cache it
            submission_flow_result = flow(**kwargs)
            if skip_caching or type(submission_flow_result) is ScoreSummary:
                return submission_flow_result
            else:
                submission = pd.DataFrame(submission_flow_result).reset_index()
                logging.info(f"flow completed, writing result to {file_path}")
                submission.to_feather(file_path)
                logging.info("finished caching")
                return submission

    return wrapper


def _create_submission_based_on_experiment(
    mlflow_run_id, technology: TechnologyRepository
):
    """
    given an mlflow experiment, run the same experiment but trained on full
    data with no cv holdout, and predict on full test input
    assumes flow has: 1) annotated kwargs 2) a `full_submission` kwarg 3) a `technology` kwarg
    TODO: should enforce this through an interface
    https://stackoverflow.com/questions/2124190/how-do-i-implement-interfaces-in-python
    """
    run = mlflow.get_run(mlflow_run_id)
    params = run.data.params
    assert params["technology"] == str(technology)
    # TODO: doesn't work if function wasn't run from __main__
    # flow_filepath = run.data.tags["mlflow.source.name"]
    # flow_filepath = "/kaggle/singlecell-kaggle/code/rbf_pca_normed_input_output.py"
    flow_filepath = params["flow_filepath"]
    flow_function = params["flow_function"]
    # import module and flow function of flow that created the model
    spec = importlib.util.spec_from_file_location(flow_function, flow_filepath)
    flow_module = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[flow_function] = flow_module
    spec.loader.exec_module(flow_module)  # type: ignore
    flow_function = getattr(flow_module, flow_function)
    # set flow to have arguments of run
    # overriding "full_submission" to be true
    kwargs = {}
    flow_kwargs = inspect.signature(flow_function).parameters
    for key in flow_kwargs.keys():
        # convert strings to necessary types
        # requires that flow has type-annotated kwargs
        kwargs[key] = flow_kwargs[key].annotation(params[key])
    kwargs["full_submission"] = True
    kwargs["technology"] = technology
    return flow_function(**kwargs)


def create_submission_from_mlflow_experiments(cite_mlflow_run_id, multi_mlflow_run_id):
    """
    public function to create a full kaggle submission based on previously run
    experiments logged into mlflow. Pattern is to 1) do experiments 2) find
    the best results in mlflow 3) take those experiment ids and input them
    into this function which will re-train the experiments will full data,
    create predictions for full test set, and submit to kaggle for scoring
    """
    c = _create_submission_based_on_experiment(cite_mlflow_run_id, cite)
    m = _create_submission_based_on_experiment(multi_mlflow_run_id, multi)
    s = SubmissionExperiments(cite_mlflow_run_id, multi_mlflow_run_id)
    return _merge_and_submit(c, m, s)


@flow
def fit_and_score_pca_input_output(
    *,
    train_inputs,
    train_targets,
    test_inputs,
    test_targets=None,
    inputs_pca_dims,
    targets_pca_dims,
    model,
    merge_inputs_for_pca=True,
):
    """
    performs model fit and score where model is predicting a reduced
    pca vector that needs to be converted back to raw data space.
    If no test_targets are passed, return the predictions instead of the score
    """
    logging = get_run_logger()
    if merge_inputs_for_pca:
        # Option to do pca on combined train & test inputs.
        pca_train_inputs, pca_test_inputs, _ = pca_inputs(  # type: ignore
            train_inputs, test_inputs, inputs_pca_dims
        )
    else:
        pca_train_inputs, _ = truncated_pca.submit(  # type: ignore
            train_inputs,
            inputs_pca_dims,
        ).result()
        pca_test_inputs, _ = truncated_pca.submit(  # type: ignore
            test_inputs,
            inputs_pca_dims,
        ).result()
    norm_pca_train_inputs = row_wise_std_scaler(pca_train_inputs).astype(np.float32)  # type: ignore
    norm_pca_test_inputs = row_wise_std_scaler(pca_test_inputs).astype(np.float32)  # type: ignore
    # TODO: center targets as they are dense anyway
    pca_train_targets, pca_model_train_targets = truncated_pca.submit(  # type: ignore
        train_targets,
        targets_pca_dims,
    ).result()
    # pca_train_targets = np.vstack([np.ones(500), np.ones(500) + 1]).T
    model.fit(norm_pca_train_inputs, pca_train_targets)
    Y_hat_train = (
        model.predict(norm_pca_train_inputs) @ pca_model_train_targets.components_  # type: ignore
    )
    train_score = correlation_score(train_targets, Y_hat_train)
    mlflow.log_metric("Train Score", train_score)
    logging.info(f"Train Score: {train_score}")
    Y_hat = model.predict(norm_pca_test_inputs) @ pca_model_train_targets.components_  # type: ignore
    if test_targets is not None:
        # TODO: review pca de-reduction
        test_score = correlation_score(test_targets, Y_hat)
        mlflow.log_metric("Test Score", test_score)
        logging.info(f"Test Score: {test_score}")
        return Score(score=test_score)
    else:
        return Y_hat


@jit
def row_corr(row_true, row_pred):
    # convert to loss, which must be minimized
    return 1 + (-1) * jnp.corrcoef(row_true, row_pred)[1, 0]


@jit
def correlation_loss(y_true, y_pred) -> float:
    """
    TODO: write unit test for this
    Loss of the predictions according to the competition rules.
    It is assumed that the predictions are not constant.
    Returns the average of each sample's Pearson correlation coefficient

    If correlation is high say .95, loss is 1 + (-.95) = .05
    If low say -.95 loss is 1 + (--.95) = 1.95
    If correlation is too high say 1.05, loss is 1 + (-1.05) = .05

    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes are different. {y_true.shape} != {y_pred.shape}")
    row_correlations = jax.vmap(row_corr, in_axes=(0, 0))(y_true, y_pred)
    return row_correlations.sum() / y_true.shape[0]


@jit
def grad_correlation_loss(y_true, y_pred) -> float:
    """
    TODO: write unit test for this
    Loss of the predictions according to the competition rules.
    It is assumed that the predictions are not constant.
    Returns the average of each sample's Pearson correlation coefficient

    If correlation is high say .95, loss is 1 + (-.95) = .05
    If low say -.95 loss is 1 + (--.95) = 1.95
    If correlation is too high say 1.05, loss is 1 + (-1.05) = .05

    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes are different. {y_true.shape} != {y_pred.shape}")

    row_correlations = jax.vmap(row_corr, in_axes=(0, 0))(y_true, y_pred)
    return row_correlations.sum() / y_true.shape[0]


# @jit
def hvp(f, inputs, vectors):
    """Hessian-vector product."""
    return jax.jvp(jax.grad(f), inputs, vectors)[1]


# See: https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
# and: https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#scikit-learn-interface
def correlation_loss_grad(train, pred):
    # TODO: review all this hessian math
    train_reshaped = train.reshape(pred.shape)
    # See https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#taking-derivatives-with-grad
    loss_function = lambda y_pred: correlation_loss(train_reshaped, y_pred)
    grad_fn = jax.grad(loss_function)
    if jnp.unique(pred).shape[0] < 2:  # type: ignore 
        # Add some noise if all predictions are the same value to avoid numerical issues.
        p = np.random.normal(size=pred.shape, scale=train.std() / 100)
        pred = pred + p
    grad = grad_fn(pred)
    # see https://towardsdatascience.com/jax-vs-pytorch-automatic-differentiation-for-xgboost-10222e1404ec
    hess = hvp(loss_function, (pred,), (jnp.ones_like(pred),))
    print(f"{hess=}")
    return grad.reshape(pred.size), jnp.ones((pred.size))  # hess.reshape(pred.size)
