import json
import os
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from sklearn_benchmarks.config import (
    BASE_LIBRARY,
    BENCHMARKING_RESULTS_PATH,
    COMPARABLE_COLS,
    PLOTLY_COLORS_TO_FILLCOLORS,
    VERSIONS_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils import (
    HoverTemplateMaker,
    find_index_nearest,
    get_library_full_name,
    is_pareto_optimal,
    make_link_to_config_file,
)


def compute_cumulated(fit_times, scores):
    """
    Return the cumulative sums of fit times and cumulated maximums of scores.

    Arguments:
    ----------
    fit_times -- sequence of fit times
    scores -- sequence of scores

    Returns:
    --------
    cumulated_fit_times -- sequence of cumulative sums of fit times
    best_val_score_so_far -- sequence of cumulated maximums of scores
    """

    cumulated_fit_times = fit_times.cumsum()
    best_val_score_so_far = pd.Series(scores).cummax()

    return cumulated_fit_times, best_val_score_so_far


def boostrap_fit_times(
    fit_times,
    scores,
    grid_scores,
    n_bootstraps=10_000,
):
    """
    Boostrap n_boostraps times the (fit_time, score) raw data points.

    Arguments:
    ---------
    fit_times -- sequence of fit times
    scores -- sequence of scores
    grid_scores -- evenly spaced scores between baseline and common maximum score

    Keyword arguments:
    ---------
    n_bootstraps -- number of bootstraps to perform (default 10 000)

    Returns:
    --------
    all_fit_times (shape (n_boostraps, grid_scores.shape[0])) -- all the boostraped fit times
    """

    all_fit_times = []
    rng = np.random.RandomState(0)
    n_samples = fit_times.shape[0]

    for _ in range(n_bootstraps):
        indices = rng.randint(n_samples, size=n_samples)
        # Compute cumulative sum of randomly selected fit times and comulative maximum of of randomly selected scores.
        cum_fit_times_permutated, cum_scores_permutated = compute_cumulated(
            fit_times.iloc[indices], scores.iloc[indices]
        )
        # Generate linear interpolation for cum_scores_permutated and cum_fit_times_permutated.
        # When max score is reached, we add nan values.
        grid_fit_times = np.interp(
            grid_scores,
            cum_scores_permutated,
            cum_fit_times_permutated,
            right=np.nan,
        )
        all_fit_times.append(grid_fit_times)

    return all_fit_times


@dataclass
class HpoBenchmarkResult:
    """
    Class responsible for storing formatted data of HPO benchmark results.
    """

    estimator: str
    lib: str
    legend: str
    color: str
    df: pd.DataFrame
    mean_grid_times: np.ndarray
    first_quartile_grid_times: np.ndarray
    third_quartile_grid_times: np.ndarray


@dataclass
class HpoBenchmarkResults:
    """
    Class responsible for storing all formatted data of HPO benchmark results and common metrics.
    """

    grid_scores: np.ndarray
    max_grid_score: float
    threshold_speedup: float
    results: List[HpoBenchmarkResult] = field(default_factory=list)

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    # Returns base lib (scikit-learn) results
    @property
    def base(self):
        return next(result for result in self.results if result.lib == BASE_LIBRARY)


def add_points_to_scatter(
    fig,
    df,
    legend,
    color,
    legendgroup,
    duration_suffix="fit",
    score_suffix="predict",
):
    mean_duration_column = f"mean_duration_{duration_suffix}"
    score_column = f"accuracy_score_{score_suffix}"

    df = df.sort_values(mean_duration_column)

    hover_template_maker = HoverTemplateMaker(df)

    fig.add_trace(
        go.Scatter(
            x=df[mean_duration_column],
            y=df[score_column],
            mode="markers",
            name=legend,
            hovertemplate=hover_template_maker.make_template(),
            customdata=hover_template_maker.make_data(),
            marker=dict(color=color),
            legendgroup=legendgroup,
        )
    )

    # Add front pareto line.
    is_pareto_column = f"is_pareto_{duration_suffix}"
    df_pareto = df[[mean_duration_column, score_column]].copy()
    df_pareto[is_pareto_column] = df_pareto.apply(
        is_pareto_optimal, args=(df_pareto,), axis=1, raw=True
    )
    df_pareto = df_pareto.query(f"{is_pareto_column} == True")
    fig.add_trace(
        go.Scatter(
            x=df_pareto[mean_duration_column],
            y=df_pareto[score_column],
            mode="lines",
            showlegend=False,
            marker=dict(color=color),
            legendgroup=legendgroup,
        )
    )


class HPOReporting:
    """
    Class responsible for running a HPO reporting for estimators specified in the configuration file.
    """

    def __init__(self, config_file=None):
        self.config_file = config_file
        config = get_full_config(config=self.config_file)
        self.config = config["hpo_reporting"]

    def set_versions(self):
        with open(VERSIONS_PATH) as json_file:
            self.versions = json.load(json_file)

    def prepare_data(self):
        """
        Prepare the raw data for reporting.

        Set benchmark_results attribute.
        """
        all_results = []
        estimators = self.config["estimators"]

        baseline_score, max_score = 0.7, 1.0
        grid_scores = np.linspace(baseline_score, max_score, 1000)

        max_grid_score = 0.0
        best_score_worst_performer = 1.0
        for estimator, params in estimators.items():
            lib = params["lib"]

            df = pd.read_csv(f"{BENCHMARKING_RESULTS_PATH}/{lib}_{estimator}.csv")

            legend = params.get("legend", lib)
            legend += f" ({self.versions[get_library_full_name(lib)]})"

            color = params["color"]

            fit_times = df.query("function == 'fit'")["mean_duration"]
            scores = df.query("function == 'predict'")["accuracy_score"]

            # The best score of the worst performing library is the minimum of maximum scores.
            best_score_worst_performer = min(best_score_worst_performer, scores.max())

            bootstrapped_fit_times = boostrap_fit_times(
                fit_times,
                scores,
                grid_scores,
            )

            # We catch warnings due to nan values in arrays.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                mean_grid_times = np.nanmean(
                    bootstrapped_fit_times,
                    axis=0,
                )

                first_quartile_grid_times = np.nanpercentile(
                    bootstrapped_fit_times,
                    25,
                    axis=0,
                )

                third_quartile_grid_times = np.nanpercentile(
                    bootstrapped_fit_times,
                    75,
                    axis=0,
                )

            # As grid_times arrays will only contain nan values at some point, we retrieve the index of the last non-nan
            # value as it corresponds to the max score obtained.
            idx_max_grid_time = mean_grid_times[~np.isnan(mean_grid_times)].shape[0]
            curr_max_grid_score = grid_scores[idx_max_grid_time]
            # We look for the maximum grid score across all librairies.
            max_grid_score = max(max_grid_score, curr_max_grid_score)

            result = HpoBenchmarkResult(
                estimator,
                lib,
                legend,
                color,
                df,
                mean_grid_times,
                first_quartile_grid_times,
                third_quartile_grid_times,
            )

            all_results.append(result)

        # We are intested in computed speedups at 95% of the worst performing library best score.
        threshold_speedup = round(best_score_worst_performer * 0.95, 3)
        self.benchmark_results = HpoBenchmarkResults(
            grid_scores,
            max_grid_score,
            threshold_speedup,
            all_results,
        )

    def scatter(self, func="fit"):
        """
        Display a scatter plot of raw results with cumulated fit or predict times in the x-axis and validation scores in the y-axis.
        """

        fig = go.Figure()

        for index, benchmark_result in enumerate(self.benchmark_results):
            df = benchmark_result.df
            df_predictions = df.query("function == 'predict'")

            comparable_columns = [col for col in COMPARABLE_COLS if col in df.columns]
            merged_columns = [
                "parameters_digest",
                "dataset_digest",
                *comparable_columns,
            ]

            if benchmark_result.lib == BASE_LIBRARY:
                df_predictions_onnx = pd.read_csv(
                    f"{BENCHMARKING_RESULTS_PATH}/onnx_{benchmark_result.estimator}.csv"
                )
                # Results are merged based on the set of parameters and the dataset used during benchmark.
                df_predictions = df_predictions.merge(
                    df_predictions_onnx[merged_columns],
                    on=["parameters_digest", "dataset_digest"],
                    how="inner",
                    suffixes=["", "_onnx"],
                )

            df_merged = df_predictions.merge(
                df.query("function == 'fit'")[merged_columns],
                on=["parameters_digest", "dataset_digest"],
                how="inner",
                suffixes=["_predict", "_fit"],
            )

            df_merged = df_merged.drop(
                ["function", "estimator", "parameters_digest", "dataset_digest"],
                axis=1,
            )

            df_merged = df_merged.dropna(axis=1)
            df_merged = df_merged.round(6)

            add_points_to_scatter(
                fig,
                df_merged,
                benchmark_result.legend,
                benchmark_result.color,
                index,
                duration_suffix=func,
                score_suffix="predict",
            )

            # For scikit-learn, we repeat the process to add ONNX prediction results
            if func == "predict" and benchmark_result.lib == BASE_LIBRARY:
                add_points_to_scatter(
                    fig,
                    df_merged,
                    f"ONNX ({self.versions['onnx']})",
                    "lightgray",
                    len(self.benchmark_results),
                    duration_suffix="onnx",
                    score_suffix="onnx",
                )

        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)

        x_title = f"{func.capitalize()} times in seconds"
        y_title = "Validation score"

        fig["layout"]["xaxis{}".format(1)]["title"] = x_title
        fig["layout"]["yaxis{}".format(1)]["title"] = y_title

        fig.show()

    def smoothed_curves(self):
        """
        Display HPO smoothed curves. Main lines are the means of bootstraped fit times.
        """

        fig = go.Figure()
        grid_scores = self.benchmark_results.grid_scores

        for index, benchmark_result in enumerate(self.benchmark_results):
            # Add mean line.
            fig.add_trace(
                go.Scatter(
                    x=benchmark_result.mean_grid_times,
                    y=grid_scores,
                    mode="lines",
                    name=benchmark_result.legend,
                    marker=dict(color=benchmark_result.color),
                    hovertemplate="Cumulated fit time: %{x:.2f}<br>Validation score: %{y:.3f}<extra></extra>",
                    legendgroup=index,
                )
            )

            # Add fill between first and third quartiles.
            for quartile_grid_times in [
                benchmark_result.first_quartile_grid_times,
                benchmark_result.third_quartile_grid_times,
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=quartile_grid_times,
                        y=grid_scores,
                        fill="tonexty",
                        showlegend=False,
                        mode="none",
                        fillcolor=PLOTLY_COLORS_TO_FILLCOLORS[benchmark_result.color],
                        hoverinfo="skip",
                        legendgroup=index,
                    )
                )

        # Add horizontal line at the score at which we compute barplot speedups.
        threshold = self.benchmark_results.threshold_speedup
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            annotation_text=f"Threshold of {threshold}",
            annotation_position="bottom right",
            annotation_font_size=12,
            fillcolor="grey",
            annotation_font_color="grey",
        )

        y_min = grid_scores.min()
        y_max = self.benchmark_results.max_grid_score
        y_max += (y_max - y_min) * 0.1  # Add a small offset for readability.
        fig.update_yaxes(range=[y_min, y_max])

        fig.update_layout(height=600, hovermode="closest")

        fig["layout"]["xaxis{}".format(1)]["title"] = "Cumulated fit times in seconds"
        fig["layout"]["yaxis{}".format(1)]["title"] = "Validation score"

        fig.show()

    def speedup_barplot(self):
        """
        Display speedup barplots at the best score of the worst performing library.
        """

        fig = go.Figure()

        grid_scores = self.benchmark_results.grid_scores
        threshold = self.benchmark_results.threshold_speedup

        base_fit_times = self.benchmark_results.base.mean_grid_times
        idx_closest_to_threshold = find_index_nearest(grid_scores, threshold)
        base_time = base_fit_times[idx_closest_to_threshold]

        df_threshold = pd.DataFrame(columns=["speedup", "legend", "color"])
        for benchmark_result in self.benchmark_results:
            idx_closest_to_threshold = find_index_nearest(grid_scores, threshold)
            lib_time = benchmark_result.mean_grid_times[idx_closest_to_threshold]
            speedup = base_time / lib_time
            row = dict(
                speedup=speedup,
                legend=benchmark_result.legend,
                color=benchmark_result.color,
            )
            df_threshold = df_threshold.append(row, ignore_index=True)

        fig.add_trace(
            go.Bar(
                x=df_threshold["legend"],
                y=df_threshold["speedup"],
                marker_color=df_threshold["color"],
                showlegend=False,
                hovertemplate="Speedup over scikit-learn: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(height=600, title=f"At validation score of {threshold}")

        fig["layout"]["xaxis{}".format(1)]["title"] = "Library"
        fig["layout"]["yaxis{}".format(1)]["title"] = f"Speedup over scikit-learn"

        fig.show()

    def speedup_curves(self):
        fig = go.Figure()
        grid_scores = self.benchmark_results.grid_scores

        for benchmark_result in self.benchmark_results:
            speedup_grid_times = (
                self.benchmark_results.base.mean_grid_times
                / benchmark_result.mean_grid_times
            )
            fig.add_trace(
                go.Scatter(
                    x=grid_scores,
                    y=speedup_grid_times,
                    mode="lines",
                    name=benchmark_result.legend,
                    marker=dict(color=benchmark_result.color),
                    hovertemplate="Validation score: %{x:.2f}<br>Cumulated fit time: %{y:.3f}<br><extra></extra>",
                )
            )

        x_min = grid_scores.min()
        x_max = self.benchmark_results.max_grid_score
        fig.update_xaxes(range=[x_min, x_max])

        fig.update_layout(height=600, hovermode="closest")

        fig["layout"]["xaxis{}".format(1)]["title"] = "Validation score"
        fig["layout"]["yaxis{}".format(1)]["title"] = "Cumulated fit times in seconds"

        fig.show()

    def make_report(self):
        self.set_versions()

        self.prepare_data()

        description = (
            "Here, we assume there is no perfect match of hyperparameters between librairies. "
            "We perform a randomized hyperparameters search with a fixed time budget for each library."
        )
        display(Markdown(f"> {description}"))

        display(Markdown("## Benchmark results for `fit`"))
        display(Markdown("### Raw fit times vs. validation scores"))
        self.scatter(func="fit")

        display(Markdown("### Smoothed HPO Curves"))
        description = (
            "We boostrap 10 000 times the hyperparameters optimization data points represented on the plot above. "
            "Then we compute the average cumulated time to reach a specific validation score by taking "
            "the mean across the bootstrapped samples. The shaded areas represent the quartiles of the bootstrap samples. "
            "The fastest libraries are therefore the closest to the upper left corner."
        )

        # Add a link to the configuration file hosted on Github.
        link_configuration_file = make_link_to_config_file(self.config_file)
        description += f" The specification of the HP grid can be found in the [configuration file]({link_configuration_file})."

        display(Markdown(f"> {description}"))
        self.smoothed_curves()

        display(Markdown("### Speedup barplot"))
        description = "The speedup threshold is found by taking 95% of the best score of the worst performing library."
        display(Markdown(f"> {description}"))
        self.speedup_barplot()

        display(Markdown("### Speedup curves"))
        self.speedup_curves()

        display(Markdown("## Benchmark results for `predict`"))
        display(Markdown("### Raw predict times vs. validation scores"))
        self.scatter(func="predict")
