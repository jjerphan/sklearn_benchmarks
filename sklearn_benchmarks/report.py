import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import qgrid
from IPython.display import HTML, Markdown, display
from plotly.subplots import make_subplots

from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    DEFAULT_COMPARE_COLS,
    ENV_INFO_PATH,
    PLOT_HEIGHT_IN_PX,
    PROFILING_RESULTS_PATH,
    REPORTING_FONT_SIZE,
    SPEEDUP_COL,
    STDEV_SPEEDUP_COL,
    TIME_REPORT_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils.plotting import (
    _gen_coordinates_grid,
    _make_hover_template,
    _order_columns,
)


class Reporting:
    """
    Runs reporting for specified estimators.
    """

    def __init__(self, config_file_path=None):
        self.config_file_path = config_file_path

    def _print_time_report(self):
        df = pd.read_csv(str(TIME_REPORT_PATH), index_col="algo")
        display(df)

    def _print_env_info(self):
        with open(ENV_INFO_PATH) as json_file:
            data = json.load(json_file)

        print(json.dumps(data, indent=2))

    def _get_estimator_default_hyperparameters(self, estimator):
        splitted_path = estimator.split(".")
        module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
        estimator_class = getattr(importlib.import_module(module), class_name)
        estimator_instance = estimator_class()
        hyperparameters = estimator_instance.__dict__.keys()
        return hyperparameters

    def _get_estimator_hyperparameters(self, estimator_config):
        if "hyperparameters" in estimator_config:
            return estimator_config["hyperparameters"].keys()
        else:
            return self._get_estimator_default_hyperparameters(
                estimator_config["estimator"]
            )

    def run(self):
        config = get_full_config(config_file_path=self.config_file_path)
        reporting_config = config["reporting"]
        benchmarking_estimators = config["benchmarking"]["estimators"]

        display(Markdown("## Time report"))
        self._print_time_report()

        reporting_estimators = reporting_config["estimators"]
        for name, params in reporting_estimators.items():
            params["n_cols"] = reporting_config["n_cols"]
            params["estimator_hyperparameters"] = self._get_estimator_hyperparameters(
                benchmarking_estimators[name]
            )
            display(Markdown(f"## {name} vs {params['against_lib']}"))
            report = Report(**params)
            report.run()

        display(Markdown("## Environment information"))
        self._print_env_info()


class Report:
    """
    Runs reporting for one estimator.
    """

    def __init__(
        self,
        name="",
        against_lib="",
        split_bars=[],
        compare=[],
        estimator_hyperparameters={},
        n_cols=None,
    ):
        self.name = name
        self.against_lib = against_lib
        self.split_bars = split_bars
        self.compare = compare
        self.n_cols = n_cols
        self.estimator_hyperparameters = estimator_hyperparameters

    def _get_benchmark_df(self, lib=BASE_LIB):
        benchmarking_results_path = str(BENCHMARKING_RESULTS_PATH)
        file_path = f"{benchmarking_results_path}/{lib}_{self.name}.csv"
        return pd.read_csv(file_path)

    def _get_compare_cols(self):
        return [*self.compare, *DEFAULT_COMPARE_COLS]

    def _make_reporting_df(self):
        base_lib_df = self._get_benchmark_df()
        base_lib_time = base_lib_df[SPEEDUP_COL]
        base_lib_std = base_lib_df[STDEV_SPEEDUP_COL]

        against_lib_df = self._get_benchmark_df(lib=self.against_lib)
        against_lib_time = against_lib_df[SPEEDUP_COL]
        against_lib_std = against_lib_df[STDEV_SPEEDUP_COL]

        compare_cols = self._get_compare_cols()

        suffixes = map(lambda lib: f"_{lib}", [BASE_LIB, self.against_lib])
        merged_df = pd.merge(
            base_lib_df,
            against_lib_df[compare_cols],
            left_index=True,
            right_index=True,
            suffixes=suffixes,
        )

        merged_df["speedup"] = base_lib_time / against_lib_time
        merged_df["stdev_speedup"] = merged_df["speedup"] * (
            np.sqrt(
                (base_lib_std / base_lib_time) ** 2
                + (against_lib_std / against_lib_time) ** 2
            )
        )

        return merged_df

    def _make_profiling_link(self, components, lib=BASE_LIB):
        function, hyperparams_digest, dataset_digest = components
        path = f"profiling/{lib}_{function}_{hyperparams_digest}_{dataset_digest}.html"
        if os.environ.get("RESULTS_BASE_URL") is not None:
            base_url = os.environ.get("RESULTS_BASE_URL")
        else:
            base_url = "http://localhost:8000/results/"
        return f"<a href='{base_url}{path}' target='_blank'>See</a>"

    def _make_plot_title(self, df):
        title = ""
        params_cols = [
            param
            for param in self.estimator_hyperparameters
            if param not in self.split_bars
            and param not in self._get_shared_hyperpameters().keys()
        ]
        values = df[params_cols].values[0]
        for index, (param, value) in enumerate(zip(params_cols, values)):
            title += "%s: %s" % (param, value)
            if index != len(list(enumerate(zip(params_cols, values)))) - 1:
                title += "<br>"
        return title

    def _print_table(self):
        df = self._make_reporting_df()
        df = df.round(3)
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        cols_to_drop = [
            col for col in cols_to_drop if col in self.estimator_hyperparameters
        ]
        df = df.drop(cols_to_drop, axis=1)
        for lib in [BASE_LIB, self.against_lib]:
            df[f"{lib}_profiling"] = df[
                ["function", "hyperparams_digest", "dataset_digest"]
            ].apply(self._make_profiling_link, lib=lib, axis=1)
        df = df.drop(["hyperparams_digest", "dataset_digest"], axis=1)
        display(HTML(df.to_html(escape=False)))

    def _make_x_plot(self, df):
        return [f"({ns}, {nf})" for ns, nf in df[["n_samples", "n_features"]].values]

    def _should_split_n_samples_train(self, df):
        return np.any(df[["n_samples", "n_features"]].duplicated().values)

    def _get_split_cols(self, df):
        split_cols = []
        if self.split_bars:
            split_cols = self.split_bars
        elif self._should_split_n_samples_train(df):
            split_cols = ["n_samples_train"]
        return split_cols

    def _add_bar_to_plotly_fig(
        self, fig, row, col, df, color="dodgerblue", name="", showlegend=False
    ):
        bar = go.Bar(
            x=self._make_x_plot(df),
            y=df["speedup"],
            name=name,
            marker_color=color,
            hovertemplate=_make_hover_template(df),
            customdata=df[_order_columns(df)].values,
            showlegend=showlegend,
            text=df["function"],
            textposition="auto",
        )
        fig.add_trace(
            bar,
            row=row,
            col=col,
        )

    def _get_shared_hyperpameters(self):
        merged_df = self._make_reporting_df()
        ret = {}
        for col in self.estimator_hyperparameters:
            unique_vals = merged_df[col].unique()
            if unique_vals.size == 1:
                ret[col] = unique_vals[0]
        return ret

    def _plot(self):
        merged_df = self._make_reporting_df()
        if self.split_bars:
            group_by_params = [
                param
                for param in self.estimator_hyperparameters
                if param not in self.split_bars
            ]
        else:
            group_by_params = "hyperparams_digest"
        print(group_by_params)
        merged_df_grouped = merged_df.groupby(group_by_params)

        n_plots = len(merged_df_grouped)
        n_rows = n_plots // self.n_cols + n_plots % self.n_cols
        coordinates = _gen_coordinates_grid(n_rows, self.n_cols)

        subplot_titles = [self._make_plot_title(df) for _, df in merged_df_grouped]

        fig = make_subplots(
            rows=n_rows,
            cols=self.n_cols,
            subplot_titles=subplot_titles,
        )

        for (row, col), (_, df) in zip(coordinates, merged_df_grouped):
            df = df.sort_values(by=["function", "n_samples", "n_features"])
            df = df.dropna(axis="columns")
            df = df.drop(["hyperparams_digest", "dataset_digest"], axis=1)
            df = df.round(3)

            split_cols = self._get_split_cols(df)
            if split_cols:
                for split_col in split_cols:
                    split_col_vals = df[split_col].unique()
                    for index, split_val in enumerate(split_col_vals):
                        filtered_df = df[df[split_col] == split_val]
                        filtered_df = filtered_df.sort_values(
                            by=["function", "n_samples", "n_features"]
                        )
                        self._add_bar_to_plotly_fig(
                            fig,
                            row,
                            col,
                            filtered_df,
                            color=px.colors.qualitative.Plotly[index],
                            name="%s: %s" % (split_col, split_val),
                            showlegend=(row, col) == (1, 1),
                        )
            else:
                self._add_bar_to_plotly_fig(fig, row, col, df)

        for i in range(1, n_plots + 1):
            fig["layout"]["xaxis{}".format(i)]["title"] = "(n_samples, n_features)"
            fig["layout"]["yaxis{}".format(i)]["title"] = "Speedup"

        fig.for_each_xaxis(
            lambda axis: axis.title.update(font=dict(size=REPORTING_FONT_SIZE))
        )
        fig.for_each_yaxis(
            lambda axis: axis.title.update(font=dict(size=REPORTING_FONT_SIZE))
        )
        fig.update_annotations(font_size=REPORTING_FONT_SIZE)
        fig.update_layout(
            height=n_rows * PLOT_HEIGHT_IN_PX, barmode="group", showlegend=True
        )
        display(Markdown(f"All estimators share the following hyperparameters:"))
        df_shared_hyperparameters = pd.DataFrame.from_dict(
            self._get_shared_hyperpameters(), orient="index", columns=["value"]
        )
        display(df_shared_hyperparameters)

        fig.show()

    def run(self):
        self._plot()
        self._print_table()
