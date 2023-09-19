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


def test_SCE_props():
    tse = SingleCellExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, sce)

    assert tse.alternative_experiments == {}
    alt = SummarizedExperiment(
        assays={"counts": counts}, row_data=df_gr, col_data=col_data
    )
    tse.alternative_experiments = {"alt": alt}
    assert tse.alternative_experiments is not None

    assert tse.assays is not None
    assert tse.row_data is not None
    assert tse.col_data is not None

    assert tse.col_pairs is None
    tse.col_pairs = {"random": col_data}
    assert tse.col_pairs is not None

    with pytest.raises(Exception):
        tse.row_pairs = counts

    assert tse.row_pairs is None

    assert tse.main_experiment_name is None
    tse.main_experiment_name = "scrna-seq"
    assert tse.main_experiment_name is not None

    assert tse.reduced_dims is None
    tse.reduced_dims = {"tsnooch": np.random.rand(ncols, 4)}
    with pytest.raises(Exception):
        tse.reduced_dims = {"tsnooch": np.random.rand(ncols - 1, 4)}
    assert tse.reduced_dims is not None

    assert tse.reduced_dim_names is not None
    assert len(tse.reduced_dim_names) == 1
