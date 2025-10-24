# Repository Guidelines

## Project Structure & Module Organization
- `src/` — Python package with core logic.
  - `src/core.py` — cellular automaton state (`Universe`, `Cell`) and step logic.
  - `src/eval.py` — metrics (entropy/MI), spectral decomposition, window selection.
  - `src/ci.py` — compliance/report generation utilities (uses matplotlib).
- `docs/` — background and specification (`docs/spec.md`, `docs/theory.md`).
- `README.md` — high‑level overview.
- `tests/` — add new tests here (not present yet).

## Build, Test, and Development Commands
- Environment: Python 3.9+ recommended.
- Install deps: `pip install -r requirements.txt`
- Run library code (example):
  - `PYTHONPATH=. python - <<'PY'
from src.core import make_universe
from src.eval import timeseries
U = make_universe(N=64)
print(timeseries(U, T=16).head())
PY`
- Module execution: `PYTHONPATH=. python -m src.eval` (modules are library‑first; no CLI prompts expected).
- Tests: `PYTHONPATH=. pytest` (configured via `pyproject.toml`).
- Long‑run experiment (headless, reproducible):
  - `PYTHONPATH=. python scripts/run_experiment.py --output-dir artifacts`.
  - Or via Makefile: `make setup && make run`.

## Coding Style & Naming Conventions
- Follow PEP 8; 4‑space indentation; max line length ~88.
- Names: modules `lower_snake`, functions/vars `snake_case`, classes `PascalCase`.
- Use type hints and docstrings for public functions/classes.
- Prefer pure, side‑effect‑light functions in `src/`; keep plotting and I/O in callers.

## Testing Guidelines
- Use `pytest`; place tests under `tests/` with `test_*.py` names (e.g., `tests/test_core.py`).
- Cover core behaviors (state updates, entropy/MI functions, decomposition invariants).
- Aim for ≥80% coverage on changed code; run `pytest -q` locally before PRs.

## Commit & Pull Request Guidelines
- Commits: imperative mood with scoped prefix, e.g., `core: implement rule110`, `eval: fix entropy edge case`, `docs: update theory`.
- PRs: include summary, rationale, and validation (commands, screenshots/plots if applicable). Link issues, note API changes, and update `README.md`/`docs/` when behavior changes.

## Security & Configuration Tips
- Do not commit large data or secrets. Keep outputs in a temp or user‑provided directory.
- Use `PYTHONPATH=.` when running modules locally; pin versions in your own env if needed.
