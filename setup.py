from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    ext_modules=[Pybind11Extension(
        "mathbr", ["main.cpp", "src/ols.cpp", "src/regularized.cpp",
                   "src/econometrics.cpp", "src/time_series.cpp",
                   "src/distributions.cpp", "src/statistics.cpp", "src/hypothesis.cpp",
                   "src/evaluation.cpp", "src/distribution_elementary.cpp",
                   "src/diagnostics.cpp", "src/nonparametric.cpp",
                   "src/time_series_diagnostics.cpp", "src/survival.cpp",
                   "src/bayesian.cpp", "src/multivariate.cpp"],
        include_dirs=["include"], cxx_std=17
    )],
    cmdclass={"build_ext": build_ext},
)
