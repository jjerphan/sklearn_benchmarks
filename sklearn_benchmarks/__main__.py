"""
The main entry point. Invoke as `sklearn_benchmarks' or `python -m sklearn_benchmarks'.
"""
import json
import os
import sys
import time

import click
import joblib
import pandas as pd
from sklearn.utils._show_versions import _get_deps_info, _get_sys_info
from threadpoolctl import threadpool_info

from sklearn_benchmarks.benchmark import Benchmark
from sklearn_benchmarks.config import (
    DEFAULT_CONFIG_FILE_PATH,
    ENV_INFO_PATH,
    TIME_REPORT_PATH,
    get_full_config,
    prepare_params,
)
from sklearn_benchmarks.utils.misc import clean_results, convert


@click.command()
@click.option(
    "--append",
    "--a",
    is_flag=True,
    required=False,
    default=False,
    help="Append benchmark results to existing ones.",
)
@click.option(
    "--config",
    "--c",
    type=str,
    default=DEFAULT_CONFIG_FILE_PATH,
    help="Path to config file.",
)
@click.option(
    "--profiling",
    "--p",
    type=click.Choice(["html", "json.gz"], case_sensitive=True),
    default=["html", "json.gz"],
    multiple=True,
    help="Profiling output formats.",
)
@click.option(
    "--estimator",
    "--e",
    type=str,
    multiple=True,
    help="Estimator to benchmark.",
)
def main(append, config, profiling, estimator):
    if not append:
        clean_results()
    config = get_full_config(config)
    benchmarking_config = config["benchmarking"]
    if not "estimators" in benchmarking_config:
        return

    all_estimators = benchmarking_config["estimators"]
    selected_estimators = all_estimators
    if estimator:
        selected_estimators = {k: all_estimators[k] for k in estimator}

    time_report = pd.DataFrame(columns=["algo", "hour", "min", "sec"])
    t0 = time.perf_counter()
    for name, params in selected_estimators.items():
        # When inherit is set, we fetch params from parent estimator
        if "inherit" in params:
            curr_estimator = params["estimator"]
            params = all_estimators[params["inherit"]]
            params["estimator"] = curr_estimator

        params = prepare_params(params)

        params["random_state"] = config.get("random_state", None)
        params["profiling_output_extensions"] = profiling

        benchmark_estimator = Benchmark(**params)
        t0_ = time.perf_counter()
        benchmark_estimator.run()
        t1_ = time.perf_counter()

        time_report.loc[len(time_report)] = [name, *convert(t1_ - t0_)]
        benchmark_estimator.to_csv()

    t1 = time.perf_counter()
    time_report.loc[len(time_report)] = ["total", *convert(t1 - t0)]
    time_report.to_csv(
        str(TIME_REPORT_PATH),
        mode="w+",
        index=False,
    )

    env_info = {}
    env_info["system_info"] = _get_sys_info()
    env_info["dependencies_info"] = _get_deps_info()
    env_info["threadpool_info"] = threadpool_info()
    env_info["cpu_count"] = joblib.cpu_count(only_physical_cores=True)

    with open(ENV_INFO_PATH, "w") as outfile:
        json.dump(env_info, outfile)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clean_results()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
