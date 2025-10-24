# Repository Guidelines

## Project Structure & Module Organization
- `docs/` — specifications (`docs/spec.md`, `docs/theory.md`) 与分层设计（例如 `docs/P0-PHYSICS.md`）。
- `src/` — Python 源码。
  - `src/p0/` — 物理层（可逆 Rule 接口 + 树结构 Cell）。
- `tests/` — pytest 测试（当前 `tests/test_physics.py` 覆盖可逆性与嵌套 ring）。

## Build, Test, and Development Commands
- Environment: Python 3.9+ recommended.
- Install deps: `pip install -r requirements.txt`
- Tests: `PYTHONPATH=. pytest`。
- Makefile: `make setup`（创建虚拟环境）与 `make test`。

## Coding Style & Naming Conventions
- Follow PEP 8; 4‑space indentation; max line length ~88.
- Names: modules `lower_snake`, functions/vars `snake_case`, classes `PascalCase`.
- Use type hints and docstrings for public functions/classes.
- Prefer pure, side‑effect‑light functions in `src/`; keep plotting and I/O in callers.

## Testing Guidelines
- 使用 `pytest`；测试文件命名为 `tests/test_*.py`。
- 当前核心测试：`tests/test_p0.py` 验证可逆性与 Rule110 行为。
- 未来层级继续追加相应测试；在提交前运行 `pytest -q`。

## Commit & Pull Request Guidelines
- Commits: imperative mood with scoped prefix, e.g., `p0: add reversible ring`, `docs: add P1 design`.
- PRs: include summary, rationale, and validation (commands, screenshots/plots if applicable). Link issues, note API changes, and update `README.md`/`docs/` when behavior changes.

## Security & Configuration Tips
- Do not commit large data or secrets. Keep outputs in a temp or user‑provided directory.
- Use `PYTHONPATH=.` when running modules locally; pin versions in your own env if needed.
