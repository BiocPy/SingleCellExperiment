"""Dummy conftest.py for singlecellexperiment.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import data.mocks as sce
import pytest


@pytest.fixture
def experiments():
    return sce
