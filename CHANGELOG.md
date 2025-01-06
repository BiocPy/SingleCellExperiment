# Changelog

## Version 0.5.1 - 0.5.3

- Add wrapper class methods to combine experiments by rows or columns.
- Expand function names for readability, still backwards compatible with the older function and method names.
- Add getters and setters to replace a specific alternative experiment or reduced dimension.
- Fixed an issue with numpy arrays as slice arguments. Code now uses Biocutils's subset functions to perform these operations.


## Version 0.5.0

- chore: Remove Python 3.8 (EOL)
- precommit: Replace docformatter with ruff's formatter

## Version 0.4.7

- Fix package version issues to support Python<=3.9. Mostly related to how anndata dependencies are versioned in the MuData package discussed [here](https://github.com/scverse/mudata/issues/82).
- The package now enforces the versions of mudata, anndata and numpy that are compatible with each other.

## Version 0.4.2 - 0.4.6

- Fix issue coercing `SummarizedExperiments` to `AnnData` objects and vice-versa.
- Handling coercions when matrices are delayed arrays or backed (for `AnnData`).
- Update sphinx configuration to run snippets in the documentation.

## Version 0.4.0 to 0.4.1

This is a complete rewrite of the package, following the functional paradigm from our [developer notes](https://github.com/BiocPy/developer_guide#use-functional-discipline).

- Migrates package to the newly udpated SE/RSE classes.
- Implement combine generics on SCE.
- Reduce dependency on a number of external packages.
- Update docstrings, tests

## Version 0.3.0

This release migrates the package to a more palatable Google's Python style guide. A major modification to the package is with casing, all `camelCase` properties, methods, functions and parameters are now `snake_case`.

With respect to the classes,`SingleCellExperiment` now extends `SummarizedExperiment`. Typehints have been updated to reflect these changes.

In addition, docstrings and documentation has been updated to use sphinx's features of linking objects to their types. Sphinx now also documents private and special dunder methods (e.g. `__getitem__`, `__copy__` etc). Intersphinx has been updated to link to references from dependent packages.

Configuration for flake8, ruff and black has been added to pyproject.toml and setup.cfg to be less annoying.

Finally, pyscaffold has been updated to use "myst-parser" as the markdown compiler instead of recommonmark. As part of the pyscaffold setup, one may use pre-commits to run some of the routine tasks of linting and formatting before every commit. While this is sometimes annoying and can be ignored with `--no-verify`, it brings some consistency to the code base.

## Version 0.1

- Initial release of SCE class
- Tests
- Documentation
