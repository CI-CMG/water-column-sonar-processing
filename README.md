# Water Column Sonar Processing
Processing tool for converting Level_0 water column sonar data to Level_1 and Level_2 derived data sets as well as generating geospatial information.

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/CI-CMG/water-column-sonar-processing/test_action.yaml)
![PyPI - Implementation](https://img.shields.io/pypi/v/water-column-sonar-processing) ![GitHub License](https://img.shields.io/github/license/CI-CMG/water-column-sonar-processing) ![PyPI - Downloads](https://img.shields.io/pypi/dd/water-column-sonar-processing) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/CI-CMG/water-column-sonar-processing) ![GitHub repo size](https://img.shields.io/github/repo-size/CI-CMG/water-column-sonar-processing)

# Setting up the Python Environment
> Python 3.10.12

# Installing Dependencies
```
uv pip install --upgrade pip
uv pip install -r pyproject.toml --all-extras
```


# Pytest
```
uv run pytest tests
```
or
> pytest --cache-clear --cov=src tests/ --cov-report=xml

# Instructions
Following this tutorial:
https://packaging.python.org/en/latest/tutorials/packaging-projects/

# Pre Commit Hook
see here for installation: https://pre-commit.com/
https://dev.to/rafaelherik/using-trufflehog-and-pre-commit-hook-to-prevent-secret-exposure-edo
```
pre-commit install --allow-missing-config
```

# Linting
Ruff
https://plugins.jetbrains.com/plugin/20574-ruff

# Colab Test
https://colab.research.google.com/drive/1KiLMueXiz9WVB9o4RuzYeGjNZ6PsZU7a#scrollTo=AayVyvpBdfIZ

# Test Coverage
20241124
8 failed, 32 passed, 3 skipped, 1 warning in 6.92s
20241125
5 failed, 35 passed, 3 skipped, 1 warning in 9.71s
3 failed, 38 passed, 3 skipped, 1 warning in 7.24s

# Tag a Release
Step 1 --> increment the semantic version in the zarr_manager.py "metadata" & the "pyproject.toml"
```commandline
git tag -a v25.4.0 -m "Releasing v25.4.0"
git push origin --tags
```

# To Publish To PROD
```commandline
uv build --no-sources
#python -m twine upload --repository pypi dist/*
uv publish
```

# TODO:
add https://pypi.org/project/setuptools-scm/
for extracting the version

# Security scanning
> bandit -r water_column_sonar_processing/

# Data Debugging
Experimental Plotting in Xarray (hvPlot):
https://colab.research.google.com/drive/18vrI9LAip4xRGEX6EvnuVFp35RAiVYwU#scrollTo=q9_j9p2yXsLV

HB0707 Zoomable Cruise:
https://hb0707.s3.us-east-1.amazonaws.com/index.html


# UV Debugging
```
uv lock --check
uv lock
uv sync --extra dev
uv run pytest tests
```
