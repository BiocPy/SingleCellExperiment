from random import random

import genomicranges
import numpy as np
import pandas as pd
from biocframe import BiocFrame
import pytest
from summarizedexperiment import SummarizedExperiment

from singlecellexperiment import SingleCellExperiment
from singlecellexperiment.SingleCellExperiment import SingleCellExperiment as sce

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
row_data = BiocFrame(
    {
        "seqnames": [
            "chr1",
            "chr2",
            "chr2",
            "chr2",
            "chr1",
            "chr1",
            "chr3",
            "chr3",
            "chr3",
            "chr3",
        ]
        * 20,
        "starts": range(100, 300),
        "ends": range(110, 310),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 20,
        "score": range(0, 200),
        "GC": [random() for _ in range(10)] * 20,
    }
)

gr = genomicranges.GenomicRanges.from_pandas(row_data.to_pandas())

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SCE_creation():
    tse = SingleCellExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, sce)


def test_SCE_creation_with_alts():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": tse},
    )

    assert tse is not None
    assert isinstance(tse, sce)


def test_SCE_creation_with_alts_should_fail():
    anrows = 200
    ancols = 2
    acounts = np.random.rand(anrows, ancols)
    adf_gr = pd.DataFrame(
        {
            "seqnames": [
                "chr1",
                "chr2",
                "chr2",
                "chr2",
                "chr1",
                "chr1",
                "chr3",
                "chr3",
                "chr3",
                "chr3",
            ]
            * 20,
            "starts": range(100, 300),
            "ends": range(110, 310),
            "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 20,
            "score": range(0, 200),
            "GC": [random() for _ in range(10)] * 20,
        }
    )
    acol_data = pd.DataFrame(
        {
            "treatment": ["ChIP", "Input"],
        }
    )

    tse = SummarizedExperiment(
        assays={"counts": acounts}, row_data=adf_gr, column_data=acol_data
    )

    with pytest.raises(Exception):
        tse = SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=col_data,
            alternative_experiments={"alt": tse},
        )

def test_SCE_creation_modifications():
    rse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": rse},
    )

    assert tse is not None
    assert isinstance(tse, sce)

    with pytest.raises(Exception):
        tse.set_reduced_dimension("something", np.random.rand(ncols - 1, 4), in_place=False)

    nassay_tse = tse.set_reduced_dimension("something", np.random.rand(tse.shape[1], 4), in_place=False)

    assert nassay_tse.get_reduced_dimension_names() != tse.get_reduced_dimension_names()

    tse.set_reduced_dimension("something", np.random.rand(tse.shape[1], 4), in_place=True)
    assert nassay_tse.get_reduced_dimension_names() == tse.get_reduced_dimension_names()

def test_SCE_different_alt_names():
    rse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=pd.DataFrame(index = ["ChIP"] * 6 )
    )

    with pytest.raises(Exception):
        tse = SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=col_data,
            alternative_experiments={"alt": rse},
        )

    with pytest.raises(Exception):
        tse = SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=pd.DataFrame(index = ["ChIP", "Input"] * 3),
            alternative_experiments={"alt": rse},
        )

    with pytest.raises(Exception):
        tse = SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=pd.DataFrame(index = ["ChIP", "Input", "Input"] * 2),
            alternative_experiments={"alt": rse},
        )

def test_SCE_dims():
    embeds = np.random.rand(counts.shape[1], 4)
    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        reduced_dimensions={
            "something": embeds
        }
    )

    assert tse is not None
    assert isinstance(tse, sce)
    assert tse.get_reduced_dimension_names() == ["something"]

    tse2 = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        reduced_dims={
            "something": embeds
        }
    )

    assert tse2 is not None
    assert isinstance(tse2, sce)
    assert tse2.get_reduced_dimension_names() == ["something"]

    print(tse.get_reduced_dimension("something"), tse2.get_reduced_dimension("something"))

    assert np.allclose(tse.get_reduced_dimension("something"), tse2.get_reduced_dimension("something"))

    with pytest.raises(Exception, match="Either 'reduced_dims' or 'reduced_dimensions' should be provided, but not both."):
        SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=col_data,
            reduced_dims={
                "something": embeds
            },
            reduced_dimensions={
                "something": embeds
            }
        )
