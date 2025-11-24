# Water Column Sonar Processing

Processing tool for converting Level_0 water column sonar data to Level_1 and Level_2 derived data sets as well as
generating geospatial information.

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/CI-CMG/water-column-sonar-processing/test_action.yaml)
![PyPI - Implementation](https://img.shields.io/pypi/v/water-column-sonar-processing) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/CI-CMG/water-column-sonar-processing) ![GitHub repo size](https://img.shields.io/github/repo-size/CI-CMG/water-column-sonar-processing)

# Setting up the Python Environment

> Python 3.12.12

# Installing Dependencies

```
source .venv/bin/activate
# or ".venv\Scripts\activate" in windows

uv pip install --upgrade pip

uv sync --all-groups

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

```commandline
uv run pytest --cov=water_column_sonar_processing
```

Current status:

```commandline
Name                                                              Stmts   Miss  Cover
-------------------------------------------------------------------------------------
water_column_sonar_processing/__init__.py                             3      0   100%
water_column_sonar_processing/aws/__init__.py                         6      0   100%
water_column_sonar_processing/aws/dynamodb_manager.py                44      4    91%
water_column_sonar_processing/aws/s3_manager.py                     161     27    83%
water_column_sonar_processing/aws/s3fs_manager.py                    18      0   100%
water_column_sonar_processing/aws/sns_manager.py                     18      1    94%
water_column_sonar_processing/aws/sqs_manager.py                     17      2    88%
water_column_sonar_processing/cruise/__init__.py                      3      0   100%
water_column_sonar_processing/cruise/create_empty_zarr_store.py      43      2    95%
water_column_sonar_processing/cruise/datatree_manager.py              0      0   100%
water_column_sonar_processing/cruise/resample_regrid.py              92      7    92%
water_column_sonar_processing/geometry/__init__.py                    5      0   100%
water_column_sonar_processing/geometry/elevation_manager.py          29      1    97%
water_column_sonar_processing/geometry/geometry_manager.py           76     11    86%
water_column_sonar_processing/geometry/line_simplification.py        39      6    85%
water_column_sonar_processing/geometry/pmtile_generation.py          77     58    25%
water_column_sonar_processing/index/__init__.py                       2      0   100%
water_column_sonar_processing/index/index_manager.py                142    113    20%
water_column_sonar_processing/model/__init__.py                       2      0   100%
water_column_sonar_processing/model/zarr_manager.py                  92     11    88%
water_column_sonar_processing/process.py                             27     27     0%
water_column_sonar_processing/processing/__init__.py                  3      0   100%
water_column_sonar_processing/processing/raw_to_netcdf.py            85     24    72%
water_column_sonar_processing/processing/raw_to_zarr.py              90      4    96%
water_column_sonar_processing/utility/__init__.py                     5      0   100%
water_column_sonar_processing/utility/cleaner.py                     13      0   100%
water_column_sonar_processing/utility/constants.py                   62      0   100%
water_column_sonar_processing/utility/pipeline_status.py             42      0   100%
water_column_sonar_processing/utility/timestamp.py                    5      0   100%
-------------------------------------------------------------------------------------
TOTAL                                                              1201    298    75%
```

# Tag a Release

Step 1 --> increment the semantic version in the zarr_manager.py "metadata" & the "pyproject.toml"

```commandline
git tag -a v25.11.7 -m "Releasing v25.11.7"
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
uv sync --all-groups
uv run pre-commit install
uv lock --check
uv lock
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

Another useful trick is to turn off "gevent" to speed up debugging:

```commandline
Python > Debugger > "Gevent Compatible"
```

# Fixing windows/wsl/ubuntu/mac git compatability

> git config --global core.filemode false
> git config --global core.autocrlf true
