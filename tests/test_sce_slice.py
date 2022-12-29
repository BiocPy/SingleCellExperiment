import pytest

from singlecellexperiment import SingleCellExperiment
import numpy as np
from random import random
import pandas as pd
from genomicranges import GenomicRanges
from singlecellexperiment.SingleCellExperiment import SingleCellExperiment as sce
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

colData = pd.DataFrame({"treatment": ["ChIP", "Input"] * 3,})


def test_SCE_slice():
    tse = SingleCellExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    tse_slice = tse[0:10, 0:3]
    assert tse_slice is not None
    assert isinstance(tse_slice, sce)

    assert len(tse_slice.rowData) == 10
    assert len(tse_slice.colData) == 3

    assert tse_slice.assay("counts").shape == (10, 3)


def test_SCE_creation_with_alts_slice():
    trse = SummarizedExperiment(
        assays={"counts": counts.copy()}, rowData=df_gr.copy(), colData=colData.copy(),
    )

    tsce = SingleCellExperiment(
        assays={"counts": counts},
        rowData=df_gr,
        colData=colData,
        altExps={"alt": trse},
    )

    tsce_slice = tsce[0:10, 0:3]

    assert tsce_slice is not None
    assert isinstance(tsce_slice, sce)

    assert len(tsce_slice.rowData) == 10
    assert len(tsce_slice.colData) == 3

    assert tsce_slice.assay("counts").shape == (10, 3)
    alt_exp = tsce_slice.altExps["alt"]
    assert alt_exp.shape == (10, 3)
