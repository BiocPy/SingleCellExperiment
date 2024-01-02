from random import random

import anndata
import genomicranges
from biocframe import BiocFrame
import biocutils
import numpy as np
import pandas as pd
from mudata import MuData

import singlecellexperiment
from singlecellexperiment.SingleCellExperiment import SingleCellExperiment

import pytest

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_combine_columns(experiments):
    combined = biocutils.combine_columns(
        experiments.se_unnamed, experiments.se_unnamed_2
    )
    assert combined is not None
    assert isinstance(combined, SingleCellExperiment)
    assert len(combined.alternative_experiments) == 0
    assert len(combined.column_data["A"]) == 20


def test_relaxed_combine_columns(experiments):
    ncols = 10
    nrows = 100
    test2 = experiments.se_unnamed_2.set_assays(
        {
            "counts": np.random.poisson(lam=10, size=(nrows, ncols)),
            "normalized": np.random.normal(size=(nrows, ncols)),
        },
        in_place=False,
    )

    with pytest.raises(Exception):
        combined = biocutils.combine_columns(experiments.se_unnamed, test2)

    combined = biocutils.relaxed_combine_columns(experiments.se_unnamed, test2)
    assert combined is not None
    assert isinstance(combined, SingleCellExperiment)
    assert len(combined.alternative_experiments) == 0
    assert len(combined.row_data["A"]) == 100


def test_combine_with_alts(experiments):
    combined = biocutils.combine_columns(
        experiments.se_with_alts1, experiments.se_with_alts2
    )
    assert combined is not None
    assert isinstance(combined, SingleCellExperiment)
    print(combined)
    assert len(combined.alternative_experiments) == 1
    assert len(combined.reduced_dim_names) == 1
    assert combined.reduced_dim_names == ["PCA"]
    assert len(combined.row_data["seqnames"]) == 3
