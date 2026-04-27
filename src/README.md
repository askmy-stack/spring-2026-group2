# Source Code — spring-2026-group2

## Structure

```
src/
├── component/      ← all Python source code (models, data, features, EDA)
├── tests/          ← full test suite (pytest)
├── docs/           ← code-level documentation
└── shellscripts/   ← bash/shell utility scripts
```

## Code Standards
- All classes and functions must have docstrings and type hints
- Tests live in `src/tests/` and run with `pytest src/tests/ -v`
- Entry points: `src/component/prepare_tensors.py`, `src/component/models/run_all_models.py`
