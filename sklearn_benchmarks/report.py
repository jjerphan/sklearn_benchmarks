import importlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import HTML, Markdown, display
from plotly.subplots import make_subplots

from sklearn_benchmarks.config import (
    BASE_LIB,
    BENCHMARKING_RESULTS_PATH,
    DEFAULT_COMPARE_COLS,
    ENV_INFO_PATH,
    HPO_CURVES_COLORS,
    PLOT_HEIGHT_IN_PX,
    REPORTING_FONT_SIZE,
    SPEEDUP_COL,
    STDEV_SPEEDUP_COL,
    TIME_REPORT_PATH,
    VERSIONS_PATH,
    get_full_config,
)
from sklearn_benchmarks.utils.misc import find_nearest
from sklearn_benchmarks.utils.plotting import (
    gen_coordinates_grid,
    identify_pareto,
    make_hover_template,
    mean_bootstrapped_curve,
    order_columns,
    quartile_bootstrapped_curve,
)


def print_time_report():
    df = pd.read_csv(str(TIME_REPORT_PATH), index_col="algo")
    df = df.sort_values(by=["hour", "min", "sec"])

    display(Markdown("## Time report"))
    for index, row in df.iterrows():
        display(Markdown("**%s**: %ih %im %is" % (index, *row.values)))


def print_env_info():
    with open(ENV_INFO_PATH) as json_file:
        data = json.load(json_file)
    display(Markdown("## Benchmark environment"))
    print(json.dumps(data, indent=2))


def display_links_to_notebooks():
    if os.environ.get("RESULTS_BASE_URL") is not None:
        base_url = os.environ.get("RESULTS_BASE_URL")
    else:
        base_url = "http://localhost:8888/notebooks/"
    notebook_titles = dict(
        sklearn_vs_sklearnex="`scikit-learn` vs. `scikit-learn-intelex` (Intel® oneAPI) benchmarks",
        sklearn_vs_onnx="`scikit-learn` vs. `ONNX Runtime` (Microsoft) benchmarks",
        gradient_boosting="Gradient boosting: randomized HPO benchmarks",
    )
    file_extension = "html" if os.environ.get("RESULTS_BASE_URL") else "ipynb"
    display(Markdown("## Notebooks"))
    for file, title in notebook_titles.items():
        display(Markdown(f"[{title}]({base_url}{file}.{file_extension})"))


class Reporting:
    """
    Runs reporting for specified estimators.
    """

    def __init__(self, config=None):
        self.config = config

    def _get_estimator_default_hyperparameters(self, estimator):
        splitted_path = estimator.split(".")
        module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
        estimator_class = getattr(importlib.import_module(module), class_name)
        estimator_instance = estimator_class()
        hyperparameters = estimator_instance.__dict__.keys()
        return hyperparameters

    def _get_estimator_hyperparameters(self, estimator_config):
        if "hyperparameters" in estimator_config:
            return estimator_config["hyperparameters"]["init"].keys()
        else:
            return self._get_estimator_default_hyperparameters(
                estimator_config["estimator"]
            )

    def run(self):
        config = get_full_config(config=self.config)
        reporting_config = config["reporting"]
        benchmarking_estimators = config["benchmarking"]["estimators"]
        reporting_estimators = reporting_config["estimators"]

        with open(VERSIONS_PATH) as json_file:
            versions = json.load(json_file)

        for name, params in reporting_estimators.items():
            params["n_cols"] = reporting_config["n_cols"]
            params["estimator_hyperparameters"] = self._get_estimator_hyperparameters(
                benchmarking_estimators[name]
            )
            key_lib_version = params["against_lib"]
            key_lib_version = reporting_config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            title = f"## `{name}`: `scikit-learn` (`{versions['scikit-learn']}`) vs. `{key_lib_version}` (`{versions[key_lib_version]}`)"
            display(Markdown(title))

            report = SingleEstimatorReport(**params)
            report.run()


class SingleEstimatorReport:
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

    def _print_tables(self):
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
        splitted_dfs = [x for _, x in df.groupby(["function"])]
        for df in splitted_dfs:
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
            hovertemplate=make_hover_template(df),
            customdata=df[order_columns(df)].values,
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

        merged_df_grouped = merged_df.groupby(group_by_params)

        n_plots = len(merged_df_grouped)
        n_rows = n_plots // self.n_cols + n_plots % self.n_cols
        coordinates = gen_coordinates_grid(n_rows, self.n_cols)

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

        text = "All estimators share the following hyperparameters: "
        df_shared_hyperparameters = pd.DataFrame.from_dict(
            self._get_shared_hyperpameters(), orient="index", columns=["value"]
        )
        for i, (index, row) in enumerate(df_shared_hyperparameters.iterrows()):
            text += "`%s=%s`" % (index, *row.values)
            if i == len(df_shared_hyperparameters) - 1:
                text += "."
            else:
                text += ", "
        display(Markdown(text))

        fig.show()

    def run(self):
        self._plot()
        self._print_tables()


class ReportingHpo:
    def __init__(self, config=None):
        self.config = config

    def _set_versions(self):
        with open(VERSIONS_PATH) as json_file:
            self._versions = json.load(json_file)

    def _display_scatter(self, func="fit"):
        fig = go.Figure()

        for index, params in enumerate(self._config["estimators"]):
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)

            legend = params.get("lib")
            legend = params.get("legend", legend)

            key_lib_version = params["lib"]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            legend += f" ({self._versions[key_lib_version]})"

            df_merged = df.query("function == 'fit'").merge(
                df.query("function == 'predict'"),
                on=["hyperparams_digest", "dataset_digest"],
                how="inner",
                suffixes=["_fit", "_predict"],
            )
            df_merged = df_merged.dropna(axis=1)
            suffix_to_drop = "_predict" if func == "fit" else "_fit"
            df_merged = df_merged.rename(
                columns={"accuracy_score_predict": "accuracy_score"}
            )
            df_merged.drop(
                df_merged.filter(regex=f"{suffix_to_drop}$").columns.tolist(),
                axis=1,
                inplace=True,
            )

            color = HPO_CURVES_COLORS[index]

            df_hover = df_merged.copy()
            df_hover.columns = df_hover.columns.str.replace(f"_{func}", "")
            df_hover = df_hover.rename(
                columns={"mean": f"mean_{func}_time", "stdev": f"stdev_{func}_time"}
            )
            df_hover = df_hover[
                df_hover.columns.drop(list(df_hover.filter(regex="digest")))
            ]
            df_hover = df_hover.round(3)

            fig.add_trace(
                go.Scatter(
                    x=df_merged[f"mean_{func}"],
                    y=df_merged["accuracy_score"],
                    mode="markers",
                    name=legend,
                    hovertemplate=make_hover_template(df_hover),
                    customdata=df_hover[order_columns(df_hover)].values,
                    marker=dict(color=color),
                    legendgroup=index,
                )
            )

            data = df_merged[[f"mean_{func}", "accuracy_score"]].values
            pareto_indices = identify_pareto(data)
            pareto_front = data[pareto_indices]
            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values

            fig.add_trace(
                go.Scatter(
                    x=pareto_front[:, 0],
                    y=pareto_front[:, 1],
                    mode="lines",
                    showlegend=False,
                    marker=dict(color=color),
                    legendgroup=index,
                )
            )

        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)
        fig["layout"]["xaxis{}".format(1)][
            "title"
        ] = f"{func.capitalize()} times in seconds"
        fig["layout"]["yaxis{}".format(1)]["title"] = "Accuracy score"
        fig.show()

    def display_smoothed_curves(self):
        plt.figure(figsize=(15, 10))

        fit_times_for_max_scores = []
        for index, params in enumerate(self._config["estimators"]):
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)

            legend = params.get("lib")
            legend = params.get("legend", legend)

            key_lib_version = params["lib"]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            legend += f" ({self._versions[key_lib_version]})"

            fit_times = df[df["function"] == "fit"]["mean"]
            scores = df[df["function"] == "predict"]["accuracy_score"]

            color = HPO_CURVES_COLORS[index]

            mean_grid_times, grid_scores = mean_bootstrapped_curve(fit_times, scores)
            idx_max_score = np.argmax(grid_scores, axis=0)
            fit_time_for_max_score = mean_grid_times[idx_max_score]
            fit_times_for_max_scores.append(fit_time_for_max_score)

            plt.plot(mean_grid_times, grid_scores, c=f"tab:{color}", label=legend)

        min_fit_time_all_constant = min(fit_times_for_max_scores)
        plt.xlim(right=min_fit_time_all_constant)
        plt.xlabel("Cumulated fit times in s")
        plt.ylabel("Validation scores")
        plt.legend()
        plt.show()

    def display_speedup_barplots(self):
        other_lib_dfs = {}
        for params in self._config["estimators"]:
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)
            if params["lib"] == BASE_LIB:
                base_lib_df = df
            key = "".join(params.get("legend", params.get("lib")))
            other_lib_dfs[key] = df

        data = []
        columns = ["score"]

        base_fit_times = base_lib_df[base_lib_df["function"] == "fit"]["mean"]
        base_scores = base_lib_df[base_lib_df["function"] == "predict"][
            "accuracy_score"
        ]

        for val in self._config["speedup"]["scores"]:
            row = [val]
            for lib, df in other_lib_dfs.items():
                if lib not in columns:
                    columns.append(lib)

                fit_times = df[df["function"] == "fit"]["mean"]
                scores = df[df["function"] == "predict"]["accuracy_score"]

                assert fit_times.shape == scores.shape

                idx, _ = find_nearest(scores, val)
                other_lib_time = fit_times.iloc[idx]

                idx, _ = find_nearest(base_scores, val)
                base_time = base_fit_times.iloc[idx]

                speedup = base_time / other_lib_time

                row.append(speedup)
            data.append(row)

        speedup_df = pd.DataFrame(columns=columns, data=data)
        speedup_df = speedup_df.set_index("score")
        _, axes = plt.subplots(3, figsize=(12, 20))

        libs = list(speedup_df.columns)
        for i in range(len(libs)):
            key_lib_version = libs[i].split(" ")[0]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            libs[i] = f"{libs[i]} ({self._versions[key_lib_version]})"

        for ax, score in zip(axes, speedup_df.index.unique()):
            speedups = speedup_df.loc[score].values
            ax.bar(x=libs, height=speedups, width=0.3, color=HPO_CURVES_COLORS)
            ax.set_xlabel("Lib")
            ax.set_ylabel(f"Speedup (time sklearn / time lib)")
            ax.set_title(f"At score {score}")

        plt.tight_layout()
        plt.show()

    def display_speedup_curves(self):
        other_lib_dfs = {}
        for params in self._config["estimators"]:
            file = f"{BENCHMARKING_RESULTS_PATH}/{params['lib']}_{params['name']}.csv"
            df = pd.read_csv(file)
            if params["lib"] == BASE_LIB:
                base_lib_df = df
            else:
                key = "".join(params.get("legend", params.get("lib")))
                other_lib_dfs[key] = df

        base_fit_times = base_lib_df[base_lib_df["function"] == "fit"]["mean"]
        base_scores = base_lib_df[base_lib_df["function"] == "predict"][
            "accuracy_score"
        ]
        base_mean_grid_times, base_grid_scores = mean_bootstrapped_curve(
            base_fit_times, base_scores
        )
        base_first_quartile_fit_times, _ = quartile_bootstrapped_curve(
            base_fit_times, base_scores, 25
        )
        base_third_quartile_fit_times, _ = quartile_bootstrapped_curve(
            base_fit_times, base_scores, 75
        )
        plt.figure(figsize=(15, 10))

        base_lib_alias = self._config["version_aliases"][BASE_LIB]
        label = f"{base_lib_alias} ({self._versions[base_lib_alias]})"
        plt.plot(
            base_grid_scores,
            base_mean_grid_times / base_mean_grid_times,
            c=HPO_CURVES_COLORS[0],
            label=label,
        )

        for index, (lib, df) in enumerate(other_lib_dfs.items()):
            fit_times = df[df["function"] == "fit"]["mean"]
            scores = df[df["function"] == "predict"]["accuracy_score"]

            mean_grid_times, grid_scores = mean_bootstrapped_curve(fit_times, scores)
            speedup_mean = base_mean_grid_times / mean_grid_times

            color = HPO_CURVES_COLORS[index + 1]

            key_lib_version = lib.split(" ")[0]
            key_lib_version = self._config["version_aliases"].get(
                key_lib_version, key_lib_version
            )
            label = f"{lib} ({self._versions[key_lib_version]})"

            plt.plot(grid_scores, speedup_mean, c=f"tab:{color}", label=label)

            first_quartile_grid_times, _ = quartile_bootstrapped_curve(
                fit_times, scores, 25
            )
            speedup_first_quartile = (
                base_first_quartile_fit_times / first_quartile_grid_times
            )

            third_quartile_grid_times, _ = quartile_bootstrapped_curve(
                fit_times, scores, 75
            )
            speedup_third_quartile = (
                base_third_quartile_fit_times / third_quartile_grid_times
            )

            plt.fill_between(
                grid_scores,
                speedup_third_quartile,
                speedup_first_quartile,
                color=color,
                alpha=0.1,
            )

        plt.xlabel("Validation scores")
        plt.ylabel(f"Speedup vs. {BASE_LIB}")
        plt.legend()
        plt.show()

    def run(self):
        config = get_full_config(config=self.config)
        self._config = config["hpo_reporting"]

        self._set_versions()

        display(Markdown("## Raw fit times vs. accuracy scores"))
        self._display_scatter(func="fit")

        display(Markdown("## Raw predict times vs. accuracy scores"))
        self._display_scatter(func="predict")

        display(Markdown("## Smoothed HPO Curves"))
        display(Markdown("> The shaded areas represent boostrapped quartiles."))
        self.display_smoothed_curves()

        display(Markdown("## Speedup barplots"))
        self.display_speedup_barplots()

        display(Markdown("## Speedup curves"))
        self.display_speedup_curves()
