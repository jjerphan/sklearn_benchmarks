import importlib
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, Markdown, display
from plotly.subplots import make_subplots
from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    COMPARABLE_COLS,
    DIFF_SCORES_THRESHOLDS,
    PLOT_HEIGHT_IN_PX,
    VERSIONS_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils import (
    HoverTemplateMaker,
    generate_coordinates_grid,
    get_library_full_name,
)


def make_profiling_link(components, lib=BASE_LIB):
    """
    Return an anchor tag pointing to a profiling HTML file result.

    Links are made from the library, the function and the digests.
    """

    function, parameters_digest, dataset_digest = components
    path = f"profiling/{lib}_{function}_{parameters_digest}_{dataset_digest}.html"

    # When the env variable PUBLISHED_BASE_URL is not set, we assume that we are working locally and that a file server is running on port 8000 to serve static files.
    if os.environ.get("PUBLISHED_BASE_URL") is not None:
        base_url = os.environ.get("PUBLISHED_BASE_URL")
    else:
        base_url = "http://localhost:8000/results/"

    return f"<a href='{base_url}{path}' target='_blank'>See</a>"


def add_bar_to_barplot(
    fig,
    row,
    col,
    df,
    color="dodgerblue",
    name="",
    showlegend=False,
):
    """
    Add a bar to a Plotly figure.
    """

    x = [f"({ns}, {nf})" for ns, nf in df[["n_samples", "n_features"]].values]
    y = df["speedup"]

    # Data displayed in hovers is less exhaustive that raw data.
    df_hover = df.copy()
    df_hover = df_hover.drop(columns=["estimator", "function"])

    hover_template_maker = HoverTemplateMaker(df_hover)

    bar = go.Bar(
        x=x,
        y=y,
        name=name,
        marker_color=color,
        hovertemplate=hover_template_maker.make_template(),
        customdata=hover_template_maker.make_data(),
        showlegend=showlegend,
        text=df["function"],
        textposition="auto",
    )

    fig.add_trace(
        bar,
        row=row,
        col=col,
    )


class HpMatchReporting:
    """
    Class responsible for running a HP reporting for estimators specified in the configuration file.
    """

    def __init__(self, other_library="", config=None, log_scale=False):
        self.other_library = other_library
        self.config = config
        self.log_scale = log_scale

    def _get_estimator_default_parameters(self, estimator):
        """
        Return the list of parameters of an estimator load from the class.
        """

        splitted_path = estimator.split(".")
        module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
        estimator_class = getattr(importlib.import_module(module), class_name)
        estimator_instance = estimator_class()
        parameters = estimator_instance.__dict__.keys()

        return parameters

    def _get_estimator_parameters(self, estimator_config):
        """
        Return the list of parameters of an estimator (from configuration file if grid is explicitly specified, otherwise it loads the class and fetch its attributes).
        """

        if "parameters" in estimator_config:
            return estimator_config["parameters"]["init"].keys()
        else:
            return self._get_estimator_default_parameters(estimator_config["estimator"])

    def make_report(self):
        """
        Run the reporting script for estimators specified in configuration file.
        """

        config = get_full_config(config=self.config)
        reporting_config = config["hp_match_reporting"]
        benchmarking_estimators = config["benchmarking"]["estimators"]
        reporting_estimators = reporting_config["estimators"][self.other_library]

        with open(VERSIONS_PATH) as json_file:
            versions = json.load(json_file)

        description = (
            "We assume here there is a perfect match between the hyperparameters of both librairies. "
            f"For a given set of parameters and a given dataset, we compute the speedup `time scikit-learn / time {self.other_library}`. "
            f"For instance, a speedup of 2 means that {self.other_library} is twice as fast as scikit-learn for a given set of parameters and a given dataset."
        )
        display(Markdown(f"> {description}"))

        for name, params in reporting_estimators.items():
            params["n_cols"] = reporting_config["n_cols"]
            params["estimator_parameters"] = self._get_estimator_parameters(
                benchmarking_estimators[name]
            )
            params["log_scale"] = self.log_scale
            other_library = params["other_library"]
            other_library = get_library_full_name(other_library)

            title = f"## `{name}`"
            subtitle = f"**{other_library} ({versions[other_library]}) vs. scikit-learn ({versions['scikit-learn']})**"

            display(Markdown(title))
            display(Markdown(subtitle))

            report = SingleEstimatorHpMatchReporting(**params)
            report.make_report()


class SingleEstimatorHpMatchReporting:
    """
    Class responsible for running HP reporting for one estimator.
    """

    def __init__(
        self,
        name="",
        other_library="",
        split_bars_by_parameters=[],
        estimator_parameters={},
        n_cols=None,
        log_scale=False,
    ):
        self.name = name
        self.other_library = other_library
        self.split_bars_by_parameters = split_bars_by_parameters
        self.n_cols = n_cols
        self.estimator_parameters = estimator_parameters
        self.log_scale = log_scale

    def get_benchmark_df(self, lib=BASE_LIB):
        """
        Return dataframe of results loaded from file.
        """

        benchmarking_results_path = str(BENCHMARKING_RESULTS_PATH)
        file_path = f"{benchmarking_results_path}/{lib}_{self.name}.csv"

        return pd.read_csv(file_path)

    def prepare_data(self):
        """
        Merge results from both libraries for an estimator.

        Set attribute df_reporting.
        Compute speedup and standard deviation of speedup.
        """

        base_lib_df = self.get_benchmark_df()
        base_lib_time = base_lib_df["mean_duration"]
        base_lib_std = base_lib_df["mean_duration"]

        other_library_df = self.get_benchmark_df(lib=self.other_library)
        other_library_time = other_library_df["mean_duration"]
        other_library_std = other_library_df["std_duration"]

        comparable_columns = [
            col for col in COMPARABLE_COLS if col in other_library_df.columns
        ]

        suffixes = map(lambda lib: f"_{lib}", [BASE_LIB, self.other_library])

        df_reporting = pd.merge(
            base_lib_df,
            other_library_df[comparable_columns],
            left_index=True,
            right_index=True,
            suffixes=suffixes,
        )

        df_reporting["speedup"] = base_lib_time / other_library_time
        df_reporting["std_speedup"] = df_reporting["speedup"] * (
            np.sqrt(
                (base_lib_std / base_lib_time) ** 2
                + (other_library_std / other_library_time) ** 2
            )
        )

        self.df_reporting = df_reporting

    def make_subplot_title(self, df):
        """
        Return the title of one subplot based on results in dataframe.
        """

        title = ""
        # We remove columns contained in split_bars_by_parameters and shared parameters because they will be displayed elsewhere on the plot.
        params_cols = [
            param
            for param in self.estimator_parameters
            if param not in self.split_bars_by_parameters
            and param not in self.get_shared_parameters().keys()
        ]
        # We add parameters with single value to the title.
        values = df[params_cols].values[0]

        for index, (param, value) in enumerate(zip(params_cols, values)):
            title += "%s: %s" % (param, value)
            if index != len(list(enumerate(zip(params_cols, values)))) - 1:
                title += "<br>"

        return title

    def print_df_raw_results(self):
        """
        Display dataframe of raw results.
        """

        df = self.df_reporting

        # We remove parameters with single value because they are displayed in the title.
        n_unique_values = df.apply(pd.Series.nunique)
        columns_to_drop = n_unique_values[n_unique_values == 1].index
        columns_to_drop = [
            col for col in columns_to_drop if col in self.estimator_parameters
        ]
        df = df.drop(columns_to_drop, axis=1)

        df = df.dropna(axis=1)
        df = df.round(3)

        # We add profiling links to df.
        for lib in [BASE_LIB, self.other_library]:
            df[f"{lib}_profiling"] = df[
                ["function", "parameters_digest", "dataset_digest"]
            ].apply(make_profiling_link, lib=lib, axis=1)

        df = df.drop(["parameters_digest", "dataset_digest"], axis=1)

        dfs = [x for _, x in df.groupby(["function"])]

        for df in dfs:
            display(HTML(df.to_html(escape=False)))

    def get_shared_parameters(self):
        """
        Return the list of parameters whose values are shared across all results.
        """

        df = self.df_reporting

        shared_params = {}
        for col in self.estimator_parameters:
            unique_vals = df[col].unique()
            if unique_vals.size == 1:
                shared_params[col] = unique_vals[0]

        return shared_params

    def plot(self):
        """
        Display speedup barplots.
        """

        df_reporting = self.df_reporting

        # By default, we group results by set of parameters, represented by the digest, and function.
        if self.split_bars_by_parameters:
            group_by_parameters = [
                parameter
                for parameter in self.estimator_parameters
                if parameter not in self.split_bars_by_parameters
            ]
        else:
            group_by_parameters = ["parameters_digest"]

        df_reporting_grouped = df_reporting.groupby(["function", *group_by_parameters])

        n_plots = len(df_reporting_grouped)
        n_rows = n_plots // self.n_cols + n_plots % self.n_cols
        coordinates = generate_coordinates_grid(n_rows, self.n_cols)

        subplot_titles = [self.make_subplot_title(df) for _, df in df_reporting_grouped]

        fig = make_subplots(
            rows=n_rows,
            cols=self.n_cols,
            subplot_titles=subplot_titles,
        )

        for (row, col), (_, df) in zip(coordinates, df_reporting_grouped):
            df = df.sort_values(by=["function", "n_samples", "n_features"])
            df = df.dropna(axis=1)
            df = df.drop(["parameters_digest", "dataset_digest"], axis=1)
            df = df.round(3)

            # We add bars for each value in the columns we want to split the bars by.
            if self.split_bars_by_parameters:
                for parameter in self.split_bars_by_parameters:
                    unique_values = df[parameter].unique()
                    for index, value in enumerate(unique_values):
                        df_filtered = df.query(f"{parameter} == {value}")
                        df_filtered = df_filtered.sort_values(
                            by=["function", "n_samples", "n_features"]
                        )
                        add_bar_to_barplot(
                            fig,
                            row,
                            col,
                            df_filtered,
                            color=px.colors.qualitative.Plotly[index],
                            name="%s: %s" % (parameter, value),
                            showlegend=(row, col) == (1, 1),
                        )
            else:
                add_bar_to_barplot(
                    fig,
                    row,
                    col,
                    df,
                )

        # Set axis titles for each plot.
        for i in range(1, n_plots + 1):  # All plots are flatten.
            fig["layout"]["xaxis{}".format(i)]["title"] = "(n_samples, n_features)"

            y_title = "Speedup"
            if self.log_scale:
                y_title += " in log scale"
            fig["layout"]["yaxis{}".format(i)]["title"] = y_title

        fig.update_layout(
            height=n_rows * PLOT_HEIGHT_IN_PX, barmode="group", showlegend=True
        )

        if self.log_scale:
            for row, col in coordinates:
                fig.update_yaxes(type="log", row=row, col=col)

        self.display_shared_parameters()

        fig.show()

    def display_shared_parameters(self):
        """
        Display list of shared parameters across all results.
        """

        df_shared_parameters = pd.DataFrame.from_dict(
            self.get_shared_parameters(), orient="index", columns=["value"]
        )

        text = "All estimators share the following parameters: "

        for i, (index, row) in enumerate(df_shared_parameters.iterrows()):
            text += "`%s=%s`" % (index, *row.values)
            if i == len(df_shared_parameters) - 1:
                text += "."
            else:
                text += ", "

        display(Markdown(text))

    def check_scores_are_close(self):
        """
        Display a HTML warning when we observe differences above the thresholds between score columns.
        """

        df_filtered = self.df_reporting.copy()

        # We find stored scores from column names.
        scores = [col for col in df_filtered.columns if "score" in col]
        scores = set(list(map(lambda score: "_".join(score.split("_")[:-1]), scores)))

        for score in scores:
            # Compute difference.
            df_filtered[f"diff_{score}s"] = np.absolute(
                df_filtered[f"{score}_{BASE_LIB}"]
                - df_filtered[f"{score}_{self.other_library}"]
            )
            df_filtered = df_filtered.query("function == 'predict'")

            threshold = DIFF_SCORES_THRESHOLDS[score]

            # Filter rows where difference between scores is above threshold.
            df_filtered = df_filtered.query(f"diff_{score}s >= {threshold}")

        if not df_filtered.empty:
            n_mismatches = len(df_filtered)
            n_total_predictions = len(self.df_reporting.query("function == 'predict'"))

            proportion_mismatches = n_mismatches / n_total_predictions * 100
            proportion_mismatches = round(proportion_mismatches, 2)

            plural_suffix = "s" if len(scores) > 1 else ""

            string_observed_diffs = "The observed differences can be found in the"
            for index, score in enumerate(scores):
                string_observed_diffs += f" diff_{score}s"
                if index == len(scores) - 1:
                    string_observed_diffs += f" column{plural_suffix}. "
                else:
                    string_observed_diffs += ", "

            string_chosen_thresholds = f"The chosen difference threshold{plural_suffix} {'are' if len(scores) > 1 else 'is'}"
            for index, score in enumerate(scores):
                threshold = DIFF_SCORES_THRESHOLDS[score]
                string_chosen_thresholds += f" {threshold} for {score}"
                if index == len(scores) - 1:
                    string_chosen_thresholds += "."
                else:
                    string_chosen_thresholds += ", "

            display(
                HTML(
                    "<div style='padding: 20px; background-color: #f44336; color: white; margin-bottom: 15px;'>"
                    "<strong>WARNING!</strong> "
                    f"Mismatch between validation scores for {n_mismatches} prediction{plural_suffix} ({proportion_mismatches}%). "
                    f"{string_observed_diffs}"
                    f"{string_chosen_thresholds}"
                    " See details in the dataframe below. "
                    "</div>"
                )
            )

            display(df_filtered)

    def make_report(self):
        self.prepare_data()

        display(Markdown("### Speedup barplots"))
        self.plot()

        self.check_scores_are_close()

        display(Markdown("### Raw results"))
        self.print_df_raw_results()
