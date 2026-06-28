"""Run the lightweight test subset when pytest is unavailable.

This intentionally supports only plain test functions and the ``tmp_path``
fixture used by NanoChat's local CPU-friendly tests.
"""

import importlib
import inspect
import tempfile
from pathlib import Path


TEST_MODULES = (
    "tests.test_engine",
    "tests.test_zh_experiment",
    "tests.test_bilingual_eval",
    "tests.test_experiment",
)


def main():
    total = 0
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        for name, function in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(function):
                continue
            parameters = list(inspect.signature(function).parameters)
            if not parameters:
                function()
            elif parameters == ["tmp_path"]:
                function(Path(tempfile.mkdtemp(prefix="nanochat-test-")))
            else:
                raise RuntimeError(
                    f"Unsupported test fixture for {module_name}.{name}: {parameters}"
                )
            total += 1
    print(f"Lightweight test runner passed {total} tests")


if __name__ == "__main__":
    main()
