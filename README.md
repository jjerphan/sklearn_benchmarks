<p align="center">
  <h3 align="center">sklearn_benchmarks</h3>

  <p align="center">
    A comparative benchmarking tool for scikit-learn's estimators
    <br />
  </p>
</p>

## Table of content

<ol>
  <li><a href="#about-the-project">About</a></li>
  <li><a href="#getting-started">Getting Started</a></li>
  <li><a href="#usage">Usage</a></li>
</ol>

## About

sklearn_benchmarks is a framework to benchmark [scikit-learn](https://github.com/scikit-learn/scikit-learn)'s estimators against concurrent implementations.

To consult benchmark results, see notebooks [scikit_learn_intelex_vs_scikit_learn.ipynb](https://github.com/mbatoul/sklearn_benchmarks/blob/master/scikit_learn_intelex_vs_scikit_learn.ipynb), [onnx_vs_scikit_learn.ipynb](https://github.com/mbatoul/sklearn_benchmarks/blob/master/onnx_vs_scikit_learn.ipynb) and [gradient_boosting.ipynb](https://github.com/mbatoul/sklearn_benchmarks/blob/master/gradient_boosting.ipynb).

For information regarding the benchmark environment and execution, see notebook [index.ipynb](https://github.com/mbatoul/sklearn_benchmarks/blob/master/index.ipynb).

These notebooks automatically deployed to `github-pages`. See current results [here](https://mbatoul.github.io/sklearn_benchmarks/).

sklearn_benchmarks is used through a command line as described below.

So far, the concurrent libraries available are:

- [Intel® oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL)
- [lightgbm](https://lightgbm.readthedocs.io/en/latest/index.html) (gradient boosting library)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) (gradient boosting library)
- [Catboost](https://catboost.ai/) (gradient boosting library)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (inferencing and training accelerator)

Available scikit-learn's estimators are:

- [KNeighborsClassifier - Brute force and KD Tree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## Getting Started

In order to setup the environment, you need to have `conda` installed. See instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

To get a local copy up and running follow these simple example steps:

```sh
$ git clone -b master --single-branch https://github.com/mbatoul/sklearn_benchmarks
$ cd sklearn_benchmarks
$ conda env create --file environment.yml
$ conda activate sklbench
$ pip install .
$ sklearn_benchmarks
# or
$ sklbench
```

## Usage

```sh
Usage: sklbench [OPTIONS]

Options:
  --append, --a                   Append benchmark results to existing ones.
  --run_profiling, --p            Activate profiling of functions with
                                  Viztracer.
  --fast, --f                     Activate the fast benchmark option for
                                  debugging purposes. Datasets will be small.
                                  All benchmarks will be converted to HPO.HPO
                                  time budget will be set to 5 seconds.
  --config, --c TEXT              Path to config file.
  --estimator, --e TEXT           Estimator to benchmark.
  --hpo_time_budget, --htb INTEGER
                                  Custom time budget for HPO benchmarks in
                                  seconds. Will be applied for all libraries.
  --help                          Show this message and exit.
```
