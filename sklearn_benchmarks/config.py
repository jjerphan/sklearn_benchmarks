import os
from pathlib import Path

import yaml

RESULTS_PATH = os.environ.get("RESULTS_PATH")
if RESULTS_PATH is None:
    RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"
else:
    RESULTS_PATH = Path(RESULTS_PATH)
PROFILING_RESULTS_PATH = RESULTS_PATH / "profiling"
BENCHMARKING_RESULTS_PATH = RESULTS_PATH / "benchmarking"
DASK_LOG_DIR = RESULTS_PATH / "dask_distributed_logs"
TIME_REPORT_PATH = RESULTS_PATH / "time_report.csv"
ENV_INFO_PATH = RESULTS_PATH / "env_info.txt"
VERSIONS_PATH = RESULTS_PATH / "versions.txt"
TIME_LAST_RUN_PATH = RESULTS_PATH / "time_most_recent_run.txt"

DEFAULT_CONFIG = "config.yml"
BASE_LIBRARY = "sklearn"
FUNC_TIME_BUDGET = 30
BENCH_LIBS = [
    "scikit-learn",
    "scikit-learn-intelex",
    "xgboost",
    "lightgbm",
    "catboost",
    "onnx",
]
HPO_PREDICTIONS_TIME_BUDGET = 3
# n_executions for each benchmarking method.
BENCHMARKING_METHODS_N_EXECUTIONS = {"hp_match": 10, "hpo": 1}
HPO_TIME_BUDGET = 600
PROFILING_OUTPUT_EXTENSIONS = ["html", "json.gz"]
JOBLIB_CACHEDIR = "tmp"
JOBLIB_VERBOSE = 1

NOTEBOOKS_TITLES = dict(
    scikit_learn_intelex_vs_scikit_learn="scikit-learn-intelex (Intel® oneAPI) vs. scikit-learn benchmarks",
    onnx_vs_scikit_learn="ONNX Runtime (Microsoft) vs. scikit-learn benchmarks",
    gradient_boosting="Gradient boosting: randomized HPO benchmarks",
)

PLOT_HEIGHT_IN_PX = 550
# The columns we want to compare between libraries. Correspond to the data stored in BenchmarkMeasurements.
COMPARABLE_COLS = [
    "mean_duration",
    "std_duration",
    "accuracy_score",
    "adjusted_rand_score",
    "r2_score",
]
# When the following thresholds are reached, HTML warnings are displayed in reporting.
DIFF_SCORES_THRESHOLDS = {
    "accuracy_score": 0.001,
    "r2_score": 0.001,
    "adjusted_rand_score": 0.01,
}
PLOTLY_COLORS_TO_FILLCOLORS = dict(
    blue="rgba(0, 0, 255, 0.1)",
    red="rgba(255, 0, 0, 0.1)",
    green="rgba(41, 124, 29, 0.1)",
    purple="rgba(178, 0, 255, 0.1)",
    orange="rgba(255, 153, 0, 0.1)",
)

SLURM_QUEUE = "normal,parietal"
SLURM_ENV_EXTRA = [
    "source /scratch/mbatoul/miniconda3/etc/profile.d/conda.sh",
    "conda activate sklbench",
]

GITHUB_BASE_URL = "https://github.com/mbatoul/sklearn_benchmarks/"


def get_full_config(config=DEFAULT_CONFIG):
    with open(config, "r") as config_file:
        config = yaml.full_load(config_file)

    return config


def parse_parameters(params):
    """Parse the parameters to get a proper representation.

    Motives: pyyaml does not support YAML 1.2 yet, hence
    numbers stored using scientific notations might be loaded
    as strings.

    PR to track: https://github.com/yaml/pyyaml/issues/486
    """

    from sklearn_benchmarks.utils import is_scientific_notation

    init_parameters = params.get("parameters", {}).get("init", {})
    for key, value in init_parameters.items():
        if not isinstance(value, list):
            continue
        for i, el in enumerate(value):
            if is_scientific_notation(el):
                init_parameters[key][i] = float(el) if "-" in el else int(float(el))

    datasets = params.get("datasets", [])
    for dataset in datasets:
        dataset["n_features"] = int(float(dataset["n_features"]))
        for i, ns_train in enumerate(dataset["n_samples_train"]):
            dataset["n_samples_train"][i] = int(float(ns_train))
        for i, ns_test in enumerate(dataset["n_samples_test"]):
            dataset["n_samples_test"][i] = int(float(ns_test))

    return params
