# Water Column Sonar Processing
Processing tool for converting L0 data to L1 and L2 as well as generating geospatial information

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/CI-CMG/water-column-sonar-processing/test_action.yaml?color=black) ![PyPI - Implementation](https://img.shields.io/pypi/v/water-column-sonar-processing?color=black) ![GitHub License](https://img.shields.io/github/license/CI-CMG/water-column-sonar-processing?color=black) ![PyPI - Downloads](https://img.shields.io/pypi/dd/water-column-sonar-processing?color=black) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/CI-CMG/water-column-sonar-processing?color=black) ![GitHub repo size](https://img.shields.io/github/repo-size/CI-CMG/water-column-sonar-processing?color=black)

# Setting up the Python Environment
> Python 3.10.12

# MacOS Pyenv Installation Instructions
  1. Install pyenv (https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv)
     1. ```brew update```
     2. ```arch -arm64 brew install pyenv```
     3. In ~/.bashrc add
        1. ```export PYENV_ROOT="$HOME/.pyenv"```
        2. ```export PATH="$PYENV_ROOT/bin:$PATH"```
        3. ```eval "$(pyenv init -)"```
     4. ```arch -arm64 brew install openssl readline sqlite3 xz zlib tcl-tk```
  2. Install pyenv-virtualenv (https://github.com/pyenv/pyenv-virtualenv)
     1. ```arch -arm64 brew install pyenv-virtualenv```
     2. In ~/.bashrc add
         1. ```eval "$(pyenv virtualenv-init -)"```
  3. Open a new terminal
  4. Install Python version
     1. ```env CONFIGURE_OPTS='--enable-optimizations' arch -arm64 pyenv install 3.10.12```
  5. Create virtual env (to delete 'pyenv uninstall 3.10.12/water-column-sonar-processing')
     1. ```pyenv virtualenv 3.10.12 water-column-sonar-processing```
  6. Set local version of python (if not done already)
     1. change directory to root of project
     2. ```pyenv local 3.10.12 water-column-sonar-processing```
     3. ```pyenv activate water-column-sonar-processing```

# Setting up IntelliJ

  1. Install the IntelliJ Python plugin
  2. Set up pyenv
     1. File -> Project Structure or CMD + ;
     2. SDKs -> + -> Add Python SDK -> Virtual Environment
     3. Select Existing Environment
     4. Choose ~/.pyenv/versions/mocking_aws/bin/python
  3. Set up Python Facet (not sure if this is required)
     1. File -> Project Structure or CMD + ;
     2. Facets -> + -> Python 
     3. Set interpreter 

# Installing Dependencies
```
uv pip install --upgrade pip
#uv pip install -r requirements_dev.txt
uv pip install -r pyproject.toml --extra dev
```


# Pytest
```commandline
uv run pytest tests
#pytest --disable-warnings
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
git tag -a v25.1.5 -m "Releasing version v25.1.5"
git push origin --tags
```

# To Publish To PROD
```commandline
python -m build
python -m twine upload --repository pypi dist/*
```

# TODO:
add https://pypi.org/project/setuptools-scm/
for extracting the version

# Security scanning
> bandit -r water_column_sonar_processing/

# Data Debugging
Experimental Plotting in Xarray (hvPlot):
https://colab.research.google.com/drive/18vrI9LAip4xRGEX6EvnuVFp35RAiVYwU#scrollTo=q9_j9p2yXsLV

HB0707 Cruise zoomable:
https://hb0707.s3.us-east-1.amazonaws.com/index.html
