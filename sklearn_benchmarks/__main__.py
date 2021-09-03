"""
The main entry point. Invoke as `sklearn_benchmarks' or `python -m sklearn_benchmarks'.
"""
import json
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import click
import joblib
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from sklearn.utils import check_random_state
from sklearn.utils._show_versions import _get_deps_info, _get_sys_info
from threadpoolctl import threadpool_info

from sklearn_benchmarks.benchmarking import Benchmark
from sklearn_benchmarks.config import (
    BENCH_LIBS,
    BENCHMARKING_RESULTS_PATH,
    DASK_LOG_DIR,
    DEFAULT_CONFIG,
    ENV_INFO_PATH,
    PROFILING_RESULTS_PATH,
    SLURM_ENV_EXTRA,
    SLURM_QUEUE,
    TIME_LAST_RUN_PATH,
    TIME_REPORT_PATH,
    VERSIONS_PATH,
    get_full_config,
    parse_parameters,
)
from sklearn_benchmarks.utils import clean_results, convert


@click.command()
@click.option(
    "--distributed",
    "--d",
    is_flag=True,
    required=False,
    default=False,
    help="Run distributed benchmarks on SLURM cluster.",
)
@click.option(
    "--append",
    "--a",
    is_flag=True,
    required=False,
    default=False,
    help="Append benchmark results to existing ones.",
)
@click.option(
    "--run_profiling",
    "--p",
    is_flag=True,
    required=False,
    default=False,
    help="Activate profiling of functions with Viztracer.",
)
@click.option(
    "--fast",
    "--f",
    is_flag=True,
    required=False,
    default=False,
    help=(
        "Activate the fast benchmark option for debugging purposes. "
        "Datasets will be small. "
        "All benchmarks will be converted to HPO."
        "HPO time budget will be set to 5 seconds."
    ),
)
@click.option(
    "--config",
    "--c",
    type=str,
    default=DEFAULT_CONFIG,
    help="Path to config file.",
)
@click.option(
    "--estimator",
    "--e",
    type=str,
    multiple=True,
    help="Estimator to benchmark.",
)
@click.option(
    "--benchmarking_method",
    "--bm",
    type=str,
    multiple=True,
    help="Select estimators by benchmarking methods.",
)
@click.option(
    "--hpo_time_budget",
    "--htb",
    type=int,
    multiple=False,
    help="Custom time budget for HPO benchmarks in seconds. Will be applied for all libraries.",
)
def main(
    distributed,
    append,
    run_profiling,
    fast,
    config,
    estimator,
    benchmarking_method,
    hpo_time_budget,
):
    """
    Run the benchmarks for all estimators specified in the configuration file.
    """

    # Store bench libs versions
    versions = {}
    for lib in BENCH_LIBS:
        versions[lib] = version(lib)
    with open(VERSIONS_PATH, "w") as outfile:
        json.dump(versions, outfile)

    # Store bench environment information
    environment_information = {}
    environment_information["system"] = _get_sys_info()
    environment_information["dependencies"] = _get_deps_info()
    environment_information["threadpool"] = threadpool_info()
    environment_information["cpu_count"] = joblib.cpu_count(only_physical_cores=True)
    with open(ENV_INFO_PATH, "w") as outfile:
        json.dump(environment_information, outfile)

    # Store current time
    with open(TIME_LAST_RUN_PATH, "w") as outfile:
        outfile.write(datetime.now().strftime("%A %d %B, %Y at %H:%M:%S"))

    if not append:
        clean_results()

    config = get_full_config(config)
    benchmarking_config = config["benchmarking"]
    if not "estimators" in benchmarking_config:
        return

    all_estimators = benchmarking_config["estimators"]
    selected_estimators = all_estimators
    if estimator:
        selected_estimators = {k: selected_estimators[k] for k in estimator}

    if benchmarking_method:
        selected_estimators = {
            e: p
            for e, p in selected_estimators.items()
            if "benchmarking_method" in p
            and p["benchmarking_method"] in benchmarking_method
        }

    random_seed = benchmarking_config.get("random_seed", None)

    time_report = pd.DataFrame(
        columns=["estimator", "benchmarking_method", "hour", "min", "sec"]
    )

    if distributed:
        cluster = SLURMCluster(
            queue=SLURM_QUEUE,
            env_extra=SLURM_ENV_EXTRA,
            cores=1,
            processes=1,
            memory="16GB",
            log_directory=DASK_LOG_DIR,
            silence_logs="info",
        )

        client = Client(cluster)

        cluster.scale(jobs=1)

    futures = []

    t0 = time.perf_counter()
    for name, params in selected_estimators.items():
        # When inherit param is set, we fetch params from parent estimator
        if "inherit" in params:
            curr_estimator = params["estimator"]
            params = all_estimators[params["inherit"]]
            # ONNX predictions are run on scikit-learn's estimators only
            if "predict_with_onnx" in params:
                params.pop("predict_with_onnx")
            params["estimator"] = curr_estimator

        for i in range(len(params["datasets"])):
            # When common dataset name is set, we inserted the common dataset's configuration in the estimator's datasets configuration.
            common_dataset_name = params["datasets"][i].get("name", None)
            if common_dataset_name is not None:
                params["datasets"][i] = benchmarking_config["common_datasets"][
                    common_dataset_name
                ]

            # Set random_state for sample generators.
            dataset_params = params["datasets"][i]["params"]
            dataset_params["random_state"] = random_seed
            params["datasets"][i]["params"] = dataset_params

        # When fast option is enabled, we use a small dataset to make benchmarks run faster.
        if fast:
            fast_dataset = dict(
                n_features=10,
                n_samples_train=[1e3],
                n_samples_test=[1],
                params={},
            )
            for i in range(len(params["datasets"])):
                params["datasets"][i].update(fast_dataset)

        params = parse_parameters(params)

        params["random_state"] = check_random_state(random_seed)
        params["run_profiling"] = run_profiling

        # When fast option is enabled, all benchmarks will be HPO benchmarks, meaning we will not explore the whole grid and stop when time limit is reached.
        if fast:
            params["benchmarking_method"] = "hpo"
            params["time_budget"] = 5
        elif hpo_time_budget is not None:
            params["time_budget"] = hpo_time_budget

        # Creates folder to store results if they don't exist.
        Path(BENCHMARKING_RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        Path(PROFILING_RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        if distributed:
            Path(DASK_LOG_DIR).mkdir(parents=True, exist_ok=True)

        benchmark = Benchmark(**params)
        start_benchmark = time.perf_counter()
        if distributed:
            future = client.submit(benchmark.run)
            while True:
                print(f"{name} benchmark running...")
                time.sleep(1)
                if future.done():
                    break
            futures.append(future)
        else:
            benchmark.run()
        end_benchmark = time.perf_counter()

        time_report.loc[len(time_report)] = [
            name,
            params["benchmarking_method"],
            *convert(end_benchmark - start_benchmark),
        ]

    # Store bench time report
    t1 = time.perf_counter()
    time_report.loc[len(time_report)] = ["total", None, *convert(t1 - t0)]
    time_report.to_csv(
        str(TIME_REPORT_PATH),
        mode="w+",
        index=False,
    )

    if distributed:
        for future in futures:
            future.result()

        cluster.scale(0)


if __name__ == "__main__":
    main()
