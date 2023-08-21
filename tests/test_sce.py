from random import random

import genomicranges
import numpy as np
import pandas as pd
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
df_gr = pd.DataFrame(
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

gr = genomicranges.from_pandas(df_gr)

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SCE_creation():
    tse = SingleCellExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, sce)


def test_SCE_creation_with_alts():
    tse = SummarizedExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=df_gr,
        col_data=col_data,
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
        assays={"counts": acounts}, row_data=adf_gr, col_data=acol_data
    )

    with pytest.raises(Exception):
        tse = SingleCellExperiment(
            assays={"counts": counts},
            row_data=df_gr,
            col_data=col_data,
            alternative_experiments={"alt": tse},
        )
