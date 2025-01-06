from random import random

import genomicranges
from biocframe import BiocFrame
import numpy as np
import pandas as pd
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


def test_SCE_slice():
    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        reduced_dims={"random_embeds": np.random.rand(ncols, 5)},
    )

    tse_slice = tse[0:10, 0:3]
    assert tse_slice is not None
    assert isinstance(tse_slice, sce)

    assert len(tse_slice.row_data) == 10
    assert len(tse_slice.col_data) == 3

    assert tse_slice.assay("counts").shape == (10, 3)

def test_SCE_slice_with_numpy():
    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        reduced_dims={"random_embeds": np.random.rand(ncols, 5)},
    )

    tse_slice = tse[np.arange(10), 0:3]
    assert tse_slice is not None
    assert isinstance(tse_slice, sce)

    assert len(tse_slice.row_data) == 10
    assert len(tse_slice.col_data) == 3

    assert tse_slice.assay("counts").shape == (10, 3)

    tse_slice = tse[np.arange(10), np.arange(3)]
    assert tse_slice is not None
    assert isinstance(tse_slice, sce)

    assert len(tse_slice.row_data) == 10
    assert len(tse_slice.col_data) == 3

    assert tse_slice.assay("counts").shape == (10, 3)

def test_SCE_creation_with_alts_slice():
    trse = SummarizedExperiment(
        assays={"counts": counts.copy()},
        row_data=row_data.copy(),
        column_data=col_data.copy(),
    )

    tsce = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": trse},
    )

    tsce_slice = tsce[0:10, 0:3]

    assert tsce_slice is not None
    assert isinstance(tsce_slice, sce)

    assert len(tsce_slice.row_data) == 10
    assert len(tsce_slice.col_data) == 3

    assert tsce_slice.assay("counts").shape == (10, 3)
    alt_exp = tsce_slice.alternative_experiments["alt"]
    assert alt_exp.shape == (200, 3)
