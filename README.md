# Water Column Sonar Processing

Processing tool for converting Level_0 water column sonar data to Level_1 and Level_2 derived data sets as well as
generating geospatial information.

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/CI-CMG/water-column-sonar-processing/test_action.yaml)
![PyPI - Implementation](https://img.shields.io/pypi/v/water-column-sonar-processing) ![GitHub License](https://img.shields.io/github/license/CI-CMG/water-column-sonar-processing) ![PyPI - Downloads](https://img.shields.io/pypi/dd/water-column-sonar-processing) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/CI-CMG/water-column-sonar-processing) ![GitHub repo size](https://img.shields.io/github/repo-size/CI-CMG/water-column-sonar-processing)

# Setting up the Python Environment

> Python 3.12.12

# Installing Dependencies

```
source .venv/bin/activate
# or ".venv\Scripts\activate" in windows

uv pip install --upgrade pip

uv pip install -r pyproject.toml --all-extras

uv run pre-commit install
```

# Pytest

```
uv run pytest --cache-clear tests # -W ignore::DeprecationWarning
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
uv run pre-commit install --allow-missing-config
# or
uv run pre-commit install
```

# Black

https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/

```
Settings > Black
Execution mode: Package
Python Interpreter: .../.venv/bin/python
Use Black Formatter: X On Code reformat, X On Save
```

# Linting

Ruff
https://plugins.jetbrains.com/plugin/20574-ruff

# Colab Test

https://colab.research.google.com/drive/1KiLMueXiz9WVB9o4RuzYeGjNZ6PsZU7a#scrollTo=AayVyvpBdfIZ

# Test Coverage

TODO

# Tag a Release

Step 1 --> increment the semantic version in the zarr_manager.py "metadata" & the "pyproject.toml"

```commandline
git tag -a v25.8.0 -m "Releasing v25.8.0"
git push origin --tags
```

# To Publish To PROD

```
uv build --no-sources
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
uv pip install --upgrade pip
uv pip install -r pyproject.toml --all-extras
uv run pre-commit install
uv lock --check
uv sync --all-groups
uv run pytest --cache-clear tests
```

# Fixing S3FS Problems

```commandline
To enable/disa asyncio for the debugger, follow the steps:
Open PyCharm
Use Shift + Shift (Search Everywhere)
In the popup type: Registry and press Enter
Find "Registry" in the list of results and click on it.
In the new popup find python.debug.asyncio.repl line and check the respective checkbox
Press Close.
Restart the IDE.
The asyncio support will be enabled in the debugger.
```

# Fixing windows/wsl/ubuntu/mac git compatability

> git config --global core.filemode false
> git config --global core.autocrlf true
