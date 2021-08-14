import glob
import importlib
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from joblib import Memory

from sklearn_benchmarks.config import (
    COMPARABLE_COLS,
    ENV_INFO_PATH,
    RESULTS_PATH,
    TIME_MOST_RECENT_RUN_PATH,
    TIME_REPORT_PATH,
)


def print_time_report():
    df = pd.read_csv(str(TIME_REPORT_PATH), index_col="estimator")
    df = df.sort_values(by=["hour", "min", "sec"])

    df["sec"] = df["sec"].round(0)
    df[["hour", "min", "sec"]] = df[["hour", "min", "sec"]].astype(int)

    row_total, df = df.iloc[-1], df.iloc[:-1]
    total_hour, total_min, total_sec = row_total[["hour", "min", "sec"]].values
    subtitle = f"Total time elapsed: {total_hour}h {total_min}m {total_sec}s"

    display(Markdown("## Time report"))
    display(Markdown(f"{subtitle}"))
    for benchmarking_method, df in df.groupby(["benchmarking_method"]):
        benchmarking_method = benchmarking_method.replace("_", " ")
        benchmarking_method = benchmarking_method.capitalize()
        display(Markdown(f"### {benchmarking_method}"))

        df = df.drop(columns=["benchmarking_method"])
        display(df)


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
        scikit_learn_intelex_vs_scikit_learn="scikit-learn-intelex (Intel® oneAPI) vs. scikit-learn benchmarks",
        onnx_vs_scikit_learn="ONNX Runtime (Microsoft) vs. scikit-learn benchmarks",
        gradient_boosting="Gradient boosting: randomized HPO benchmarks",
    )
    file_extension = "html" if os.environ.get("RESULTS_BASE_URL") else "ipynb"
    display(Markdown("## Notebooks"))
    for file, title in notebook_titles.items():
        display(Markdown(f"### [{title}]({base_url}{file}.{file_extension})"))


def display_time_most_recent_run():
    with open(TIME_MOST_RECENT_RUN_PATH) as file:
        str_most_recent_time = file.readlines()[0]
    display(Markdown(f"> These results were obtained on {str_most_recent_time}."))


def gen_coordinates_grid(n_rows, n_cols):
    """
    Return a grid of coordinates in a 2D plan from number of rows cols.

    Example:
        gen_coordinates_grid(2, 2) => [(1, 1), (1, 2), (2, 1), (2, 2)]
    """

    coordinates = [[j for j in range(n_cols)] for _ in range(n_rows)]
    for i in range(len(coordinates)):
        for j in range(len(coordinates[0])):
            coordinates[i][j] = (i + 1, j + 1)
    coordinates = list(itertools.chain(*coordinates))

    return coordinates


def identify_pareto(data):
    """
    Return the indices of the 2D data's front pareto.

    Pareto optimality is a situation where no individual or preference criterion can be better off
    without making at least one individual or preference criterion worse off or without any loss thereof.
    The Pareto front is the set of all Pareto efficient allocations, conventionally shown graphically.
    """
    n = data.shape[0]
    all_indices = np.arange(n)
    front_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if data.iloc[i][0] > data.iloc[j][0] and data.iloc[i][1] < data.iloc[j][1]:
                front_pareto[i] = 0
                break
    return all_indices[front_pareto]


def is_pareto_optimal(point, data):
    """
    Return True if the point is Pareto optimal, False otherwise.
    """
    pareto_indices = identify_pareto(data)
    front_pareto = data.iloc[pareto_indices].to_numpy()
    point_in_front_pareto = np.any(np.all(point == front_pareto, axis=1))

    return point_in_front_pareto


_cachedir = "tmp"
memory = Memory(_cachedir, verbose=0)


@memory.cache
def gen_data(
    generator_path, n_samples=1000, n_features=10, random_state=None, **kwargs
):
    """
    Returns a tuple of data from the specified generator.
    """
    splitted_path = generator_path.split(".")
    module, func = ".".join(splitted_path[:-1]), splitted_path[-1]
    generator_func = getattr(importlib.import_module(module), func)
    data = generator_func(
        n_samples=n_samples, n_features=n_features, random_state=random_state, **kwargs
    )

    return data


def is_scientific_notation(string):
    """
    Return True if string represents a number written in scientific notation.

    Examples:
         is_scientific_notation("1e2") => True
         is_scientific_notation("100") => False
    """
    return isinstance(string, str) and bool(re.match(r"1[eE](\-)*\d{1,}", string))


def clean_results():
    """
    Remove static files generated during benchmarks.
    """
    extensions = [".csv", ".html", ".json.gz", ".txt"]
    files = []
    for extension in extensions:
        files_path = str(RESULTS_PATH / "**/*") + extension
        files += glob.glob(str(files_path), recursive=True)

    for file in files:
        if file.split("/")[-1] == "index.html":
            continue
        try:
            os.remove(file)
        except OSError as e:
            print("Error: %s : %s" % (file, e.strerror))


def convert(seconds):
    """
    Return number of hours, minutes and seconds converted from total number of seconds.
    """
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return hour, min, sec


def find_index_nearest(array, value):
    """
    Return the index of the closest element to the value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_lib_alias(lib):
    """
    Return the library's name alias.

    We often want to use the full library name instead of the abbreviation.
    """
    aliases = dict(sklearn="scikit-learn", sklearnex="scikit-learn-intelex")
    return aliases.get(lib, lib)


def string_matches_substrings(string, substrings):
    """
    Return True if a string matches one or more of the provided substrings, False otherwise.
    """
    return any(map(string.__contains__, substrings))


def diff_between_lists(l1, l2):
    """
    Return the difference between two lists.
    """
    return list(set(l1) - set(l2)) + list(set(l2) - set(l1))


def get_position(column):
    """
    Return the expected position of a dataframe column.
    """
    top_columns = [
        "estimator",
        "n_samples_train",
        "n_samples",
        "n_features",
    ]
    if column in top_columns:
        return 4
    elif "mean_duration" in column:
        return 3
    elif "std_duration" in column:
        return 2
    elif "score" in column:
        return 1
    elif "speedup" in column:
        return 0
    else:
        return -1


class HoverTemplateMaker:
    """
    Class responsible for generating hover templates for Plotly plots.

    See https://plotly.com/python/hover-text-and-formatting/#customizing-hover-text-with-a-hovertemplate.
    """

    def __init__(self, df):
        self.df = df
        self.dimensions = ["n_samples_train", "n_samples", "n_features"]

    def __len__(self):
        return len(self.dimensions) + len(self.measurements) + len(self.parameters)

    def __iter__(self):
        return iter([*self.dimensions, *self.measurements, *self.parameters])

    def at(self, i):
        return [*self.dimensions, *self.measurements, *self.parameters][i]

    def split_columns_in_groups(self):
        measurements_columns = [
            *COMPARABLE_COLS,
            "speedup",
            "std_speedup",
            "iteration_throughput",
            "latency",
        ]
        measurements = [
            col
            for col in self.df.columns
            if string_matches_substrings(
                col,
                measurements_columns,
            )
        ]

        # We want measurements to be ordered in a specific way.
        measurements = sorted(measurements, key=get_position, reverse=True)
        self.measurements = list(measurements)

        # We assume parameters are the columns that are not dimensions nor measurements.
        parameters = diff_between_lists(
            self.df.columns, [*self.dimensions, *self.measurements]
        )
        parameters = sorted(parameters)
        self.parameters = list(parameters)

    def make_template(self):
        self.split_columns_in_groups()

        # Section titles before the elements of the sections.
        titles_indices = {0: "Dimensions"}
        titles_indices[len(self.dimensions)] = "Benchmark measurements"
        titles_indices[len(self.dimensions) + len(self.measurements)] = "Parameters"

        template = ""
        for i in range(len(self)):
            if i in titles_indices:
                template += "<br>"
                template += f"<b>{titles_indices[i]}</b><br>"
            template += "%s: %%{customdata[%i]}<br>" % (self.at(i), i)
        template += "<extra></extra>"

        return template

    def make_data(self):
        return self.df[[*self.dimensions, *self.measurements, *self.parameters]].values
