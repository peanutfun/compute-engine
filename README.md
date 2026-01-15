# CRACE

Climate Risk Assessement Compute Engine

## Installation (dev)

With pip:

```
pip install -e ./
pip install --group dev
pre-commit install
```

With uv:

```
uv sync --dev
uv tool install pre-commit --with pre-commit-uv
pre-commit install
```
