# Water Column Sonar Processing
Processing tool for converting L0 data to L1 and L2 as well as generating geospatial information

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

  1. Add dependencies with versions to requirements.txt
  2. ```pip install --upgrade pip && pip install -r requirements_dev.txt```


# Pytest
```commandline
pytest --disable-warnings
```

# Instructions
Following this tutorial:
https://packaging.python.org/en/latest/tutorials/packaging-projects/

# To Publish To TEST
```commandline
python -m build
# python -m build --sdist
# python -m build --wheel
python -m twine upload --repository testpypi dist/*
pytho -m pip install --index-url https://test.pypi.org/simple/ hello-pypi-rudy-klucik
python
```
```
from water-column-sonar-processing import ZarrManager
example.add_one(2)
```

# To Publish To PROD
```commandline
python -m build
python -m twine upload --repository pypi dist/*
```

# Pre Commit Hook
https://dev.to/rafaelherik/using-trufflehog-and-pre-commit-hook-to-prevent-secret-exposure-edo
```
pre-commit install --allow-missing-config
```

# Linting
Ruff
https://plugins.jetbrains.com/plugin/20574-ruff

# Colab Test
https://colab.research.google.com/drive/1KiLMueXiz9WVB9o4RuzYeGjNZ6PsZU7a#scrollTo=AayVyvpBdfIZ