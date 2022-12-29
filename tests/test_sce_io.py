import singlecellexperiment
import numpy as np
from random import random
import pandas as pd
from genomicranges import GenomicRanges
from singlecellexperiment.SingleCellExperiment import SingleCellExperiment

import anndata
import pytest

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


def test_SCE_toAnnData():
    tse = SingleCellExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    adata = tse.toAnnData()
    assert adata is not None
    assert isinstance(adata, anndata.AnnData)


def test_SCE_fromH5AD():
    tse = singlecellexperiment.readH5AD("tests/data/adata.h5ad")

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    assert tse.assays is not None
    assert tse.rowData is not None
    assert tse.colData is not None

    # slice
    sliced = tse[0:10, 1:5]

    assert sliced is not None
    assert isinstance(sliced, SingleCellExperiment)

    assert sliced.assays is not None
    assert sliced.rowData is not None
    assert sliced.colData is not None

    assert sliced.shape == (10, 4)


def test_SCE_from10xH5():
    tse = singlecellexperiment.read10xH5("tests/data/tenx.sub.h5")

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    assert tse.assays is not None
    assert tse.rowData is not None
    assert tse.colData is None

    # slice
    sliced = tse[0:10, 1:5]

    assert sliced is not None
    assert isinstance(sliced, SingleCellExperiment)

    assert sliced.assays is not None
    assert sliced.rowData is not None
    assert sliced.colData is None

    assert sliced.shape == (10, 4)
