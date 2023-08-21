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

gr = genomicranges.fromPandas(df_gr)

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SCE_props():
    tse = SingleCellExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, sce)

    assert tse.altExps is None
    alt = SummarizedExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )
    tse.altExps = {"alt": alt}
    assert tse.altExps is not None

    assert tse.assays is not None
    assert tse.row_data is not None
    assert tse.col_data is not None

    assert tse.colPairs is None
    tse.colPairs = {"random": col_data}
    assert tse.colPairs is not None

    with pytest.raises(Exception):
        tse.rowPairs = counts

    assert tse.rowPairs is None

    assert tse.mainExperimentName is None
    tse.mainExperimentName = "scrna-seq"
    assert tse.mainExperimentName is not None

    assert tse.reduced_dims is None
    tse.reduced_dims = {"tsnooch": np.random.rand(ncols, 4)}
    with pytest.raises(Exception):
        tse.reduced_dims = {"tsnooch": np.random.rand(ncols - 1, 4)}
    assert tse.reduced_dims is not None

    assert tse.reducedDimNames is not None
    assert len(tse.reducedDimNames) == 1
