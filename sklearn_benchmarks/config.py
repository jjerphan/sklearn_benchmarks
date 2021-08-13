import os
from pathlib import Path

import yaml

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"
PROFILING_RESULTS_PATH = RESULTS_PATH / "profiling"
BENCHMARKING_RESULTS_PATH = RESULTS_PATH / "benchmarking"
TIME_REPORT_PATH = RESULTS_PATH / "time_report.csv"
ENV_INFO_PATH = RESULTS_PATH / "env_info.txt"
VERSIONS_PATH = RESULTS_PATH / "versions.txt"

DEFAULT_CONFIG = "config.yml"
BASE_LIB = "sklearn"
FUNC_TIME_BUDGET = 30
PLOT_HEIGHT_IN_PX = 350
COMPARABLE_COLS = [
    "mean_duration",
    "std_duration",
    "accuracy_score",
    "adjusted_rand_score",
    "r2_score",
]
BENCH_LIBS = [
    "scikit-learn",
    "scikit-learn-intelex",
    "xgboost",
    "lightgbm",
    "catboost",
    "onnx",
]
HPO_PREDICTIONS_TIME_BUDGET = 3
BENCHMARKING_METHODS_N_EXECUTIONS = {"hp_match": 10, "hpo": 1}
HPO_TIME_BUDGET = 600
PROFILING_OUTPUT_EXTENSIONS = ["html", "json.gz"]
DIFF_SCORES_THRESHOLDS = {
    "accuracy_score": 0.001,
    "r2_score": 0.001,
    "adjusted_rand_score": 0.001,
}


def get_full_config(config=None):
    if config is None:
        config = os.environ.get("DEFAULT_CONFIG")
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
