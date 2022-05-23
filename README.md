# PyNP4VTT

Python library providing NonParametric models for Value of Travel Time analysis.

## Installation steps

* Use `pip` to install the `py-np4vtt` library normally.
  + Recommended: do it in a fresh virtual environment
    - Create env: `python3 -m venv <chosen_venv_directory>`
    - Activate env: `source <chosen_venv_directory>/bin/activate`
  + Either install from TestPyPI
    - `python3 -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple py-np4vtt`
  + Or from the normal (public/official) PyPI
    - `python3 -m pip install py-np4vtt`

## Instructions for contributing to the project

### One-time setup

* This project uses `poetry` as its dependency management, virtualenv management and release (build) tool
   + Install following the steps described in https://python-poetry.org/docs/master/#installing-with-the-official-installer
* Setup PyPI credentials to be able to publish packages
   1. Make an account on `https://pypi.org`. Ask (optional) for invitation to become contributor on PyPI.
   2. Add API token on the "account settings" page of PyPI (global scope)
   3. Setup Poetry:
      - `poetry config pypi-token.pypi "<your_api_token>"`
* Setup TestPyPI credentials to be able to publish packages
   1. Make an account on `https://test.pypi.org`. Ask (optional) for invitation to become contributor on TestPyPI.
   2. Add API token on the "account settings" page of TestPyPI (global scope)
   3. Setup Poetry:
      - `poetry config repositories.testpypi https://test.pypi.org/legacy`
      - `poetry config pypi-token.testpypi "<your_api_token>"`

### Sometimes: update package dependencies

* It is advisable to sometimes (every couple of months) update the package's dependencies
  + Using newer versions (if possible) of dependencies gives you security fixes (sometimes also performance improvements)
* Steps:
  1. Make a backup of the lock file (in case you need to rollback the update):
     - `mv poetry.lock bkp-poetry.lock`
  2. Then create a new lock file with updated versions of dependencies, and install all fresh:
     - `poetry update --lock`
     - `poetry env remove python && poetry install`
  3. Test that the program still works as expected
  4. If the program breaks after the update, revert to the previous state by restoring the old lock file:
     - `mv bkp-poetry.lock poetry.lock`
     - `poetry env remove python && poetry install`
  5. If nothing is broken after the update, remove the old lock file:
     - `rm bkp-poetry.lock`

### Building a new version and releasing/uploading to PyPI or TestPyPI

1. Do the actual contribution to the project 🙂
2. Increment the package's version number in `pyproject.toml`
3. Build the package (wheel and source): `poetry build`. The built artifacts will be placed in the `dist` folder
4. Publish:
   + Either to PyPI: `poetry publish`
   + Or to TestPyPI: `poetry publish -r testpypi`

