from random import random

import numpy as np
import pandas as pd
from biocframe import BiocFrame

from singlecellexperiment.SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

ncols = 10
nrows = 100
se_unnamed = SingleCellExperiment(
    assays={"counts": np.random.poisson(lam=10, size=(nrows, ncols))},
    row_data=BiocFrame({"A": [1] * nrows}),
    column_data=BiocFrame({"A": [1] * ncols}),
)

se_unnamed_2 = SingleCellExperiment(
    assays={
        "counts": np.random.poisson(lam=10, size=(nrows, ncols)),
        # "normalized": np.random.normal(size=(nrows, ncols)),
    },
    row_data=BiocFrame({"A": [3] * nrows}),
    column_data=BiocFrame({"A": [3] * ncols}),
)

rowdata1 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
coldata1 = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_2", "SAM_3"],
        "disease": ["True", "True", "True"],
        "doublet_score": [0.15, 0.62, 0.18],
    },
    index=["cell_1", "cell_2", "cell_3"],
)
se_with_alts1 = SingleCellExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=rowdata1,
    column_data=coldata1,
    row_names=["HER2", "BRCA1", "TPFK"],
    column_names=["cell_1", "cell_2", "cell_3"],
    metadata={"seq_type": "paired"},
    reduced_dims={"PCA": np.random.poisson(lam=10, size=(3, 5))},
    alternative_experiments={
        "modality1": SingleCellExperiment(
            assays={"counts2": np.random.poisson(lam=10, size=(3, 3))},
        )
    },
)

rowdata2 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
coldata2 = pd.DataFrame(
    {
        "sample": ["SAM_4", "SAM_5", "SAM_6"],
        "disease": ["True", "False", "True"],
        "doublet_score": [0.05, 0.23, 0.54],
    },
    index=["cell_4", "cell_5", "cell_6"],
)
se_with_alts2 = SingleCellExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=rowdata2,
    column_data=coldata2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
    reduced_dims={"PCA": np.random.poisson(lam=5, size=(3, 5))},
    alternative_experiments={
        "modality1": SingleCellExperiment(
            assays={"counts2": np.random.poisson(lam=5, size=(3, 3))},
        )
    },
)
