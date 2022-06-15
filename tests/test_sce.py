import pytest

from singlecellexperiment import SingleCellExperiment
import numpy as np
from random import random
import pandas as pd
from genomicranges import GenomicRanges
from singlecellexperiment.SingleCellExperiment import (
    SingleCellExperiment as sce,
)
from summarizedexperiment import SummarizedExperiment

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

gr = GenomicRanges.fromPandas(df_gr)

colData = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SCE_creation():
    tse = SingleCellExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, sce)


def test_SCE_creation_with_alts():
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    tse = SingleCellExperiment(
        assays={"counts": counts},
        rowData=df_gr,
        colData=colData,
        alterExps={"alt": tse},
    )

    assert tse is not None
    assert isinstance(tse, sce)
