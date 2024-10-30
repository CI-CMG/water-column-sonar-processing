# water-column-sonar-processing
Processing tool for converting L0 data to L1 and L2 as well as generating geospatial information

## Setting up the Python Environment
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
     1. ```env PYTHON_CONFIGURE_OPTS="--enable-shared"```
     2. ```env CONFIGURE_OPTS='--enable-optimizations' arch -arm64 pyenv install 3.10.12```
  5. Create virtual env (to delete 'pyenv uninstall 3.10.12/water-column-sonar-processing')
     1. ```pyenv virtualenv 3.10.12 water-column-sonar-processing```
  6. Set local version of python (if not done already)
     1. change directory to root of project
     2. ```pyenv local 3.10.12 water-column-sonar-processing```
     3. ```pyenv activate water-column-sonar-processing```

## Setting up IntelliJ

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

## Installing Dependencies

  1. Add dependencies with versions to requirements.txt
  2. ```pip install --upgrade pip && pip install -r requirements_dev.txt```


## Pytest
```commandline
pytest --disable-warnings
```

## Security scanning for keys

> trufflehog git file://water-column-sonar-processing --only-verified

```bash
#trufflehog filesystem water-column-sonar-processing/ --only-verified
# {"chunks": 81804, "bytes": 453020391, "verified_secrets": 3, "unverified_secrets": 0, "scan_duration": "12.384960625s", "trufflehog_version": "3.82.13"}
trufflehog git file://water-column-sonar-processing/ --only-verified
# {"chunks": 674, "bytes": 498742, "verified_secrets": 0, "unverified_secrets": 0, "scan_duration": "253.11925ms", "trufflehog_version": "3.82.13"}
```