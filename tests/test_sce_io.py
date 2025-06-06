from random import random

import anndata
import genomicranges
from biocframe import BiocFrame
import numpy as np
import pandas as pd
from mudata import MuData
from scipy import sparse

import singlecellexperiment
from singlecellexperiment import SingleCellExperiment
from summarizedexperiment import SummarizedExperiment
from hdf5array import Hdf5CompressedSparseMatrix

__author__ = "jkanche, keviny2"
__copyright__ = "jkanche, keviny2"
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


def test_SCE_to_anndata():
    tse = SingleCellExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    adata = tse.to_anndata()
    assert adata is not None
    assert isinstance(adata[0], anndata.AnnData)
    assert adata[0].shape[0] == counts.shape[1]
    assert adata[0].shape[1] == counts.shape[0]

    assert adata[1] is None

def test_SCE_to_anndata_with_alts():
    se = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": se},
    )

    adata = tse.to_anndata()
    assert adata is not None
    assert adata[1] is None
    assert isinstance(adata[0], anndata.AnnData)
    assert adata[0].shape[0] == counts.shape[1]
    assert adata[0].shape[1] == counts.shape[0]

    adata = tse.to_anndata(include_alternative_experiments=True)
    assert adata is not None
    assert adata[1] is not None
    alt_rtrip = adata[1]["alt"]
    assert isinstance(alt_rtrip, anndata.AnnData)
    assert alt_rtrip.shape[0] == counts.shape[1]
    assert alt_rtrip.shape[1] == counts.shape[0]


def test_SCE_fromH5AD():
    tse = singlecellexperiment.read_h5ad("tests/data/adata.h5ad")

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    assert tse.assays is not None
    assert tse.row_data is not None
    assert tse.col_data is not None

    # slice
    sliced = tse[0:10, 1:5]

    assert sliced is not None
    assert isinstance(sliced, SingleCellExperiment)

    assert sliced.assays is not None
    assert sliced.row_data is not None
    assert sliced.col_data is not None

    assert sliced.shape == (10, 4)


def test_SCE_from10x_mtx():
    tse = singlecellexperiment.read_tenx_mtx("tests/data/raw_feature_bc_matrix")

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    assert tse.assays is not None
    assert tse.row_data is not None
    assert tse.col_data is not None

    # slice
    sliced = tse[0:10, 1:5]

    assert sliced is not None
    assert isinstance(sliced, SingleCellExperiment)

    assert sliced.assays is not None
    assert sliced.row_data is not None
    assert sliced.col_data is not None

    assert sliced.shape == (10, 4)


def test_SCE_from10xH5():
    tse = singlecellexperiment.read_tenx_h5("tests/data/tenx.sub.h5")

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    assert tse.assays is not None
    assert isinstance(tse.assay(0), Hdf5CompressedSparseMatrix)
    assert tse.row_data is not None
    assert tse.col_data is not None

    # slice
    sliced = tse[0:10, 1:5]

    assert sliced is not None
    assert isinstance(sliced, SingleCellExperiment)

    assert sliced.assays is not None
    assert sliced.row_data is not None
    assert sliced.col_data is not None

    assert sliced.shape == (10, 4)

    tse = singlecellexperiment.read_tenx_h5(
        "tests/data/tenx.sub.h5", realize_assays=True
    )
    assert isinstance(tse.assay(0), sparse.spmatrix)


def test_SCE_randomAnnData():
    np.random.seed(1)

    n, d, k = 1000, 100, 10

    z = np.random.normal(loc=np.arange(k), scale=np.arange(k) * 2, size=(n, k))
    w = np.random.normal(size=(d, k))
    y = np.dot(z, w.T)

    adata = anndata.AnnData(y)
    adata.obs_names = [f"obs_{i+1}" for i in range(n)]
    adata.var_names = [f"var_{j+1}" for j in range(d)]

    tse = singlecellexperiment.SingleCellExperiment.from_anndata(adata)

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    # to avoid unknown mapping types;
    # ran into an issue with anndata.compat._overloaded_dict.OverloadedDict when loading a h5ad
    adata.uns = {".internal": [f"obs_{i+1}" for i in range(n)]}
    tse = singlecellexperiment.SingleCellExperiment.from_anndata(adata)

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)


def test_SCE_to_mudata():
    tse = SingleCellExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data
    )

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    result = tse.to_mudata()
    assert result is not None
    assert isinstance(result, MuData)
