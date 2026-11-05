"""What a script imports, resolved rather than grepped.

The corpus scripts' "imports nothing that can open a vault" guards must not
test each import LINE for a dotted substring: `from health_advisor import
vault` never contains `health_advisor.vault`. Two checks, shared by
tests/test_corpus_fetch.py and tests/test_corpus_registry.py:

* `banned_imports(path, banned)` parses the file with `ast` and resolves every
  import statement anywhere in it (function bodies included) to the module
  names it can bind: `import a.b` binds `a` and `a.b`; `from a import b` binds
  `a` and, because `b` may be a submodule, `a.b`; `from a import *` binds
  everything under `a`. String-literal `importlib.import_module(...)` and
  `__import__(...)` calls count too.
* `banned_after_import(path, banned)` loads the script in a fresh interpreter
  (module level only, `main` never runs) and reads `sys.modules`, which sees
  what the file's imports pull in transitively. It is NOT run with `-I`: that
  would drop PYTHONPATH, and a checkout tested with `PYTHONPATH=<checkout>`
  would be checked against an installed copy of the package instead. Its
  working directory is the script's own, which is `sys.path[0]` when the
  script itself is run.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


def _within(name: str, banned: str) -> bool:
    return name == banned or name.startswith(banned + ".")


def _bound_names(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, module name) for every module an import in `tree` can bind;
    a star import is recorded as `pkg.*`."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                found.extend((node.lineno, ".".join(parts[:i])) for i in range(1, len(parts) + 1))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # No absolute name to resolve; `banned_imports` refuses it
                # outright (the corpus scripts are not packages).
                found.append((node.lineno, "." * node.level + (node.module or "")))
                continue
            parts = node.module.split(".")
            found.extend((node.lineno, ".".join(parts[:i])) for i in range(1, len(parts) + 1))
            found.extend((node.lineno, f"{node.module}.{alias.name}") for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called = (func.attr if isinstance(func, ast.Attribute)
                      else func.id if isinstance(func, ast.Name) else None)
            if called in ("import_module", "__import__") and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                found.append((node.lineno, node.args[0].value))
    return found


def banned_imports(path: Path, banned: tuple[str, ...] | list[str]) -> list[str]:
    """`"<line>: <module>"` for every import in the file that binds a banned
    module or one beneath it; a star import from a banned module's parent
    package counts, because it can bind the banned module."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for line, name in _bound_names(tree):
        if name.startswith("."):
            hits.append(f"{line}: relative import {name}")
        elif name.endswith(".*"):
            package = name[:-2]
            if any(_within(b, package) or _within(package, b) for b in banned):
                hits.append(f"{line}: {name}")
        elif any(_within(name, b) for b in banned):
            hits.append(f"{line}: {name}")
    return hits


def banned_after_import(path: Path, banned: tuple[str, ...] | list[str]) -> list[str]:
    """Banned modules present in `sys.modules` once the script has been loaded
    in a fresh, isolated interpreter (its module level only)."""
    probe = (
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location('script_under_import_guard', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe, str(path)], cwd=str(Path(path).parent),
                          capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise AssertionError(f"loading {path} failed:\n{done.stderr}")
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    return sorted(name for name in loaded if any(_within(name, b) for b in banned))


# The modules that can open health data, which the corpus scripts (network-capable
# or not) must never import. A literal here, so a script emptying its own list
# cannot make its guard vacuous.
VAULT_OPENING_MODULES = (
    "health_advisor.db", "health_advisor.vault", "health_advisor.context",
    "health_advisor.receiver", "health_advisor.normalize",
)
