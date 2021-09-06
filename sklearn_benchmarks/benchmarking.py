import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from numpy.testing import assert_allclose
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils._testing import set_random_state
from viztracer import VizTracer

from sklearn_benchmarks.config import (
    BENCHMARKING_METHODS_N_EXECUTIONS,
    BENCHMARKING_RESULTS_PATH,
    FUNC_TIME_BUDGET,
    HPO_PREDICTIONS_TIME_BUDGET,
    HPO_TIME_BUDGET,
    PROFILING_OUTPUT_EXTENSIONS,
    PROFILING_RESULTS_PATH,
    RESULTS_PATH,
)
from sklearn_benchmarks.utils import (
    generate_data,
    dynamic_load,
    load_metrics,
)


def run_benchmark_one_func(
    func,
    estimator,
    profiling_result_path,
    X,
    y=None,
    n_executions=10,
    run_profiling=False,
    onnx_model_filepath=None,
    **kwargs,
):
    """
    Run benchmark for one function.

    Arguments:
    ----------
    func -- function to run
    estimator -- instance of an estimator class
    profiling_result_path -- file path to store results of profiling
    X -- training data

    Keyword arguments:
    ------------------
    y -- target values (default None)
    n_executions -- number of executions of the function to run (default 10)
    run_profiling -- whether profiling should be run (default False)
    onnx_model_filepath -- file path to retrieve ONNX model (default None, when predictions should not be made with ONNX)
    kwargs -- keyword arguments that should be passed to benchmarked function

    Returns:
    --------
    func_result -- result of the benchmarked function (None if fit, predicted targets for predict)
    benchmark_measurements -- a BenchmarkMeasurements instance (see above for details)
    """
    if run_profiling:
        # First run with a profiler (not timed)
        with VizTracer(verbose=0) as tracer:
            tracer.start()
            if y is not None:
                func_result = func(X, y, **kwargs)
            else:
                if onnx_model_filepath is not None:
                    sess = rt.InferenceSession(onnx_model_filepath)
                    input_name = sess.get_inputs()[0].name
                    label_name = sess.get_outputs()[0].name

                    func_result = sess.run(
                        [label_name], {input_name: X.astype(np.float32)}
                    )[0]
                else:
                    func_result = func(X, **kwargs)
            tracer.stop()
            for extension in PROFILING_OUTPUT_EXTENSIONS:
                output_file = f"{profiling_result_path}.{extension}"
                tracer.save(output_file=output_file)

    # Next runs: at most n_executions runs or 30 sec total execution time
    times = []
    start_global = time.perf_counter()
    for _ in range(n_executions):
        start_iter = time.perf_counter()

        if y is not None:
            func_result = func(X, y, **kwargs)
        else:
            if onnx_model_filepath is not None:
                sess = rt.InferenceSession(onnx_model_filepath)
                input_name = sess.get_inputs()[0].name
                label_name = sess.get_outputs()[0].name

                func_result = sess.run(
                    [label_name], {input_name: X.astype(np.float32)}
                )[0]
            else:
                func_result = func(X, **kwargs)

        end_iter = time.perf_counter()
        times.append(end_iter - start_iter)

        # Benchmark for one func should not exceed FUNC_TIME_BUDGET seconds across all executions.
        if end_iter - start_global > FUNC_TIME_BUDGET:
            break

    n_iter = estimator.n_iter_ if hasattr(estimator, "n_iter_") else None

    mean_duration = np.mean(times)
    std_duration = np.std(times)
    iteration_throughput = X.nbytes / mean_duration / 1e9
    latency = mean_duration / X.shape[0]

    benchmark_measurements = BenchmarkMeasurements(
        mean_duration,
        std_duration,
        n_iter,
        iteration_throughput,
        latency,
    )

    return func_result, benchmark_measurements


class ResultPathMaker:
    """
    Class responsible for generating paths to result files (profiling and benchmarking).
    """

    def __init__(self, library, estimator, parameters_digest, dataset_digest):
        self.library = library
        self.estimator = estimator
        self.parameters_digest = parameters_digest
        self.dataset_digest = dataset_digest

    def profiling_path(self, function, library=None):
        if library is None:
            library = self.library

        return f"{PROFILING_RESULTS_PATH}/{self.library}_{function}_{self.parameters_digest}_{self.dataset_digest}"

    def benchmarking_path(self, estimator, library=None, extension="csv"):
        if library is None:
            library = self.library

        return f"{BENCHMARKING_RESULTS_PATH}/{library}_{estimator}.{extension}"


@dataclass
class BenchmarkMeasurements:
    """
    Class responsible for storing measurements made during benchmarks.
    """

    mean_duration: float
    std_duration: float
    n_iter: int
    iteration_throughput: float
    latency: float


@dataclass
class RawBenchmarkResult:
    """
    Class responsible for the result of the benchmark for one given configuration, i.e. a set of parameters and a dataset for one estimator's function.
    """

    estimator: str
    library: str
    function: str
    n_samples_train: int
    n_samples: int
    n_features: int
    parameters_digest: str
    dataset_digest: str
    benchmark_measurements: BenchmarkMeasurements
    parameters_batch: dict
    scores: Dict = field(default_factory=dict)

    def __str__(self):
        output = repr(self)
        output += "\n---"

        return output


@dataclass
class RawBenchmarkResults:
    """
    Class responsible for the results of benchmarks for one estimator.
    """

    results: List[RawBenchmarkResult] = field(default_factory=list)

    def __iter__(self):
        return iter(self.results)

    def append(self, result):
        self.results.append(result)

    def to_csv(self, csv_path):
        results = []
        for result in self:
            result = asdict(result)
            result = {
                **result,
                **result["benchmark_measurements"],
                **result["parameters_batch"],
                **result["scores"],
            }
            del result["benchmark_measurements"]
            del result["parameters_batch"]
            del result["scores"]
            results.append(result)

        pd.DataFrame(results).to_csv(
            csv_path,
            mode="w+",
            index=False,
        )


class Benchmark:
    """
    Class responsible for benchmarking one estimator.
    """

    def __init__(
        self,
        name="",
        estimator="",
        inherit=False,
        metrics=[],
        parameters={},
        datasets=[],
        predict_with_onnx=False,
        random_state=None,
        benchmarking_method="",
        time_budget=HPO_TIME_BUDGET,
        profiling_file_type="",
        run_profiling=False,
    ):
        self.name = name
        self.estimator = estimator
        self.EstimatorClass = dynamic_load(self.estimator)
        self.inherit = inherit
        self.metrics = metrics
        self.parameters = parameters
        self.datasets = datasets
        self.random_state = random_state
        self.predict_with_onnx = predict_with_onnx
        self.benchmarking_method = benchmarking_method
        self.time_budget = time_budget
        self.profiling_file_type = profiling_file_type
        self.run_profiling = run_profiling

    def make_parameters_grid(self):
        """
        Return a shuffled hyperparameters grid generated from the parameters specified in the configuration file.
        """

        init_parameters = self.parameters.get("init", {})
        if not init_parameters:
            estimator = self.EstimatorClass()
            # Parameters grid should have list values
            init_parameters = {k: [v] for k, v in estimator.__dict__.items()}
        grid = list(ParameterGrid(init_parameters))
        self.random_state.shuffle(grid)

        return grid

    def run(self):
        """
        Run the benchmark script.
        """

        library = self.estimator.split(".")[0]
        metrics = load_metrics(self.metrics)
        parameters_grid = self.make_parameters_grid()
        benchmark_results = RawBenchmarkResults()
        # If predictions should also be made with ONNX, we store them in a different object.
        if self.predict_with_onnx:
            onnx_benchmark_results = RawBenchmarkResults()
        n_executions = BENCHMARKING_METHODS_N_EXECUTIONS[self.benchmarking_method]

        start = time.perf_counter()
        for index, dataset in enumerate(self.datasets):
            n_features = dataset["n_features"]
            n_samples_train = dataset["n_samples_train"]
            n_samples_test = sorted(dataset["n_samples_test"], reverse=True)

            for ns_train in n_samples_train:
                # We set n_samples as the sum of n_samples_train and the maximum of the n_samples_test as we
                # are going to split the data in training and test data afterwards.
                n_samples = ns_train + max(n_samples_test)
                sample_generator = dataset["sample_generator"]
                sample_generator = dynamic_load(sample_generator)

                X, y = generate_data(
                    sample_generator,
                    n_samples=n_samples,
                    n_features=n_features,
                    **dataset["params"],
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    train_size=ns_train,
                    shuffle=False,
                    random_state=self.random_state,
                )

                for parameters_batch in parameters_grid:
                    estimator = self.EstimatorClass(**parameters_batch)
                    set_random_state(estimator, random_state=self.random_state)
                    # Use digests to identify results later in reporting
                    parameters_digest = joblib.hash(parameters_batch)
                    dataset_digest = joblib.hash((index, dataset))
                    result_path_maker = ResultPathMaker(
                        library,
                        self.name,
                        parameters_digest,
                        dataset_digest,
                    )

                    # Benchmark fit function first.
                    _, benchmark_measurements = run_benchmark_one_func(
                        estimator.fit,
                        estimator,
                        result_path_maker.profiling_path("fit"),
                        X_train,
                        y=y_train,
                        n_executions=n_executions,
                        run_profiling=self.run_profiling,
                    )

                    if self.predict_with_onnx:
                        initial_type = [
                            ("float_input", FloatTensorType([None, X_train.shape[1]]))
                        ]
                        onnx_model_filepath = f"{RESULTS_PATH}/{library}_{self.name}_{parameters_digest}_{dataset_digest}.onnx"

                        onx = convert_sklearn(estimator, initial_types=initial_type)

                        with open(onnx_model_filepath, "wb") as f:
                            f.write(onx.SerializeToString())

                    benchmark_result = RawBenchmarkResult(
                        self.name,
                        library,
                        "fit",
                        ns_train,
                        ns_train,
                        n_features,
                        parameters_digest,
                        dataset_digest,
                        benchmark_measurements,
                        parameters_batch,
                    )
                    print(benchmark_result)
                    benchmark_results.append(benchmark_result)

                    start_predictions = time.perf_counter()
                    for i in range(len(n_samples_test)):
                        ns_test = n_samples_test[i]
                        X_test_, y_test_ = X_test[:ns_test], y_test[:ns_test]

                        # NOTE: explicitly indicating the benchmarked function name here
                        bench_func = estimator.kneighbors
                        function_name = 'kneighbors'

                        if self.predict_with_onnx:
                            onnx_profiling_path = result_path_maker.profiling_path(
                                function_name,
                                library="onnx",
                            )

                            (
                                onnx_func_result,
                                onnx_benchmark_measurements,
                            ) = run_benchmark_one_func(
                                bench_func,
                                estimator,
                                onnx_profiling_path,
                                X_test_,
                                n_executions=n_executions,
                                run_profiling=self.run_profiling,
                                onnx_model_filepath=onnx_model_filepath,
                            )

                            onnx_scores = {}
                            for metric in metrics:
                                score = metric(y_test_, onnx_func_result)
                                onnx_scores[metric.__name__] = score

                            onnx_benchmark_result = RawBenchmarkResult(
                                self.name,
                                "onnx",
                                function_name,
                                ns_train,
                                ns_test,
                                n_features,
                                parameters_digest,
                                dataset_digest,
                                onnx_benchmark_measurements,
                                parameters_batch,
                                onnx_scores,
                            )

                            print(onnx_benchmark_result)
                            onnx_benchmark_results.append(onnx_benchmark_result)

                        func_result, benchmark_measurements = run_benchmark_one_func(
                            bench_func,
                            estimator,
                            result_path_maker.profiling_path(function_name),
                            X_test_,
                            n_executions=n_executions,
                            run_profiling=self.run_profiling,
                        )

                        scores = {}
                        for metric in metrics:
                            score = metric(y_test_, func_result)
                            scores[metric.__name__] = score

                        if self.predict_with_onnx:
                            # Check ONNX predictions consistency with scikit-learn's predictions.
                            assert onnx_func_result.shape == func_result.shape

                            for score in scores.keys():
                                assert_allclose(onnx_scores[score], scores[score])

                        benchmark_result = RawBenchmarkResult(
                            self.name,
                            library,
                            function_name,
                            ns_train,
                            ns_test,
                            n_features,
                            parameters_digest,
                            dataset_digest,
                            benchmark_measurements,
                            parameters_batch,
                            scores,
                        )

                        print(benchmark_result)
                        benchmark_results.append(benchmark_result)

                        csv_path = result_path_maker.benchmarking_path(self.name)
                        benchmark_results.to_csv(csv_path)

                        if self.predict_with_onnx:
                            csv_path = result_path_maker.benchmarking_path(
                                self.name, library="onnx"
                            )
                            onnx_benchmark_results.to_csv(csv_path)

                        if self.benchmarking_method == "hpo":
                            now = time.perf_counter()
                            if now - start > self.time_budget:
                                if self.predict_with_onnx:
                                    os.remove(onnx_model_filepath)
                                return
                            if now - start_predictions > HPO_PREDICTIONS_TIME_BUDGET:
                                break

                    if self.predict_with_onnx:
                        os.remove(onnx_model_filepath)
