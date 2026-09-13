"""Keep the printable user manual runnable and complete as the API changes."""

import pathlib
import re

import mathbr


MANUAL = pathlib.Path(__file__).parents[1] / "docs" / "mathbr_manual.tex"


def test_latex_manual_covers_public_symbols():
    document = MANUAL.read_text(encoding="utf-8")
    modules = (mathbr.activations, mathbr.losses, mathbr.distributions,
               mathbr.statistics, mathbr.hypothesis, mathbr.evaluation,
               mathbr.diagnostics, mathbr.nonparametric,
               mathbr.time_series_diagnostics, mathbr.survival,
               mathbr.bayesian)
    functions = [name for module in modules for name in dir(module)
                 if not name.startswith("__")]
    classes = [name for name in dir(mathbr)
               if isinstance(getattr(mathbr, name), type)]
    methods = {name for class_name in classes
               for name in dir(getattr(mathbr, class_name))
               if not name.startswith("_")}
    methods.update(name for name in dir(mathbr.hypothesis.TTestResult)
                   if not name.startswith("_"))
    for result_type in (mathbr.evaluation.RocCurve,
                        mathbr.evaluation.PrecisionRecallCurve,
                        mathbr.evaluation.CalibrationCurve,
                        mathbr.time_series_diagnostics.LjungBoxResult,
                        mathbr.survival.KaplanMeierResult,
                        mathbr.survival.LogRankResult,
                        mathbr.bayesian.BetaPosterior,
                        mathbr.bayesian.NormalPosterior):
        methods.update(name for name in dir(result_type)
                       if not name.startswith("_"))
    assert all(name in document for name in functions + classes + list(methods))


def test_latex_manual_python_examples_run():
    document = MANUAL.read_text(encoding="utf-8")
    blocks = re.findall(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}",
                        document, re.DOTALL)
    examples = [block for block in blocks if "import mathbr" in block]
    assert len(examples) >= 10
    for example in examples:
        exec(example, {})
