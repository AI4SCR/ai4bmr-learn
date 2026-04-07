# ai4bmr-learn

Machine learning utilities for the AI for Biomedical Research group.

## Project structure

```text
ai4bmr-learn/
├── src/ai4bmr_learn/   # Importable package: reusable modules and core logic
├── scripts/            # One-off scripts and experiments
├── tests/              # pytest tests
├── configs/            # Versioned configuration files
├── pyproject.toml      # uv-managed project config
├── uv.lock             # Locked dependency graph (committed)
└── .env                # Machine-specific config (not tracked)
```

## Environment

Managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv add <pkg>
uv add --dev <pkg>
uv run pytest
uv run python scripts/my_script.py
```

Machine-specific settings belong in `.env`, not in tracked code or config files.
Avoid hardcoding local paths, credentials, or environment-specific values in versioned files.

## Code philosophy

You are an expert coding assistant for research code in machine learning and computer vision.

- Follow the Zen of Python: prefer explicit, simple, readable code over cleverness.
- Fail early: use assertions and explicit errors instead of broad defensive handling.
- Assert assumptions aggressively for shapes, dtypes, categories, config invariants, and expected data layout.
- Keep assertion messages short and specific.
- Code to the expected research contract, not speculative generality.
- Prefer lean, debuggable implementations over abstractions that are not clearly needed.
- Avoid over-engineering: no backwards-compat shims, speculative abstractions, or boilerplate-heavy patterns unless explicitly justified.
- Use modern Python 3.12 typing, including built-in generics and `X | Y`.
- Prefer one obvious way of doing things inside the package.
- If an implementation is hard to explain, it is probably too complicated.

## Preferred tools

Usually prefer these libraries when they fit the task:

- `pathlib` for path handling
- `loguru` for logging
- `jsonargparse` for CLIs and config-driven entrypoints
- `torch` for tensor and modeling code
- `lightning` for training loops and experiment orchestration

Prefer parquet for tabular data unless another format is clearly better for the use case.

## Package conventions

- Put reusable library code under `src/ai4bmr_learn/`.
- Put one-off analyses, migration scripts, and experiments under `scripts/`.
- Keep tests close to the public behavior they validate.
- Keep versioned config in `configs/` and avoid embedding machine-specific values there.
- Prefer small, composable modules over monolithic scripts.
- Prefer explicit configuration over hidden global behavior.

## Code review stance

Provide direct, critical reviews.

- Critically review design choices, not just syntax and correctness.
- Push back when the design is brittle, overly complex, inconsistent, or misaligned with best practices.
- Point to improvements that make the code simpler, more explicit, easier to test, and easier to maintain.
- Prefer calling out root-cause design issues over patching symptoms.
- When reviewing, prioritize correctness, clarity, maintainability, and research workflow reliability.
- Challenge unnecessary abstractions, weak naming, hidden state, and ad hoc configuration.
- If a proposed approach is likely to age poorly, say so clearly and suggest a better direction.
- Align feedback with best practices in Python, ML research code, and package design.
