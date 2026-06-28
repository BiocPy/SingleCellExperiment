from random import random

import anndata
import genomicranges
import numpy as np
import pandas as pd
from biocframe import BiocFrame
from hdf5array import Hdf5CompressedSparseMatrix
from mudata import MuData
from scipy import sparse
from summarizedexperiment import SummarizedExperiment

import singlecellexperiment
from singlecellexperiment import SingleCellExperiment

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
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    adata = tse.to_anndata()
    assert adata is not None
    assert isinstance(adata[0], anndata.AnnData)
    assert adata[0].shape[0] == counts.shape[1]
    assert adata[0].shape[1] == counts.shape[0]

    assert adata[1] is None


def test_SCE_to_anndata_with_alts():
    se = SummarizedExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

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

    tse = singlecellexperiment.read_tenx_h5("tests/data/tenx.sub.h5", realize_assays=True)
    assert isinstance(tse.assay(0), sparse.spmatrix)


def test_SCE_randomAnnData():
    np.random.seed(1)

    n, d, k = 1000, 100, 10

    z = np.random.normal(loc=np.arange(k), scale=np.arange(k) * 2, size=(n, k))
    w = np.random.normal(size=(d, k))
    y = np.dot(z, w.T)

    adata = anndata.AnnData(y)
    adata.obs_names = [f"obs_{i + 1}" for i in range(n)]
    adata.var_names = [f"var_{j + 1}" for j in range(d)]

    tse = singlecellexperiment.SingleCellExperiment.from_anndata(adata)

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    # to avoid unknown mapping types;
    # ran into an issue with anndata.compat._overloaded_dict.OverloadedDict when loading a h5ad
    adata.uns = {".internal": [f"obs_{i + 1}" for i in range(n)]}
    tse = singlecellexperiment.SingleCellExperiment.from_anndata(adata)

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    # set raw
    adata.raw = adata.copy()
    tse = singlecellexperiment.SingleCellExperiment.from_anndata(adata)

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)
    assert tse.alternative_experiments is not None
    assert "raw" in tse.alternative_experiments
    assert isinstance(tse.alternative_experiments["raw"], SummarizedExperiment)
    assert tse.alternative_experiments["raw"].shape == (d, n)


def test_SCE_to_mudata():
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    assert tse is not None
    assert isinstance(tse, SingleCellExperiment)

    result = tse.to_mudata()
    assert result is not None
    assert isinstance(result, MuData)


def test_tenx_io_edge_cases():
    import os
    import tempfile

    import h5py
    import pytest

    from singlecellexperiment.io.tenx import read_tenx_h5, read_tenx_mtx

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "matrix.mtx"), "w") as f:
            f.write("%%MatrixMarket matrix coordinate real general\n% Metadata\n10 4 3\n1 1 1.0\n2 2 2.0\n3 3 3.0\n")

        with open(os.path.join(tmpdir, "barcodes.tsv"), "w") as f:
            f.write("BC1\nBC2\nBC3\nBC4\n")

        with open(os.path.join(tmpdir, "genes.tsv"), "w") as f:
            f.write(
                "G1\tGene1\nG2\tGene2\nG3\tGene3\nG4\tGene4\nG5\tGene5\nG6\tGene6\nG7\tGene7\nG8\tGene8\nG9\tGene9\nG10\tGene10\n"
            )

        sce = read_tenx_mtx(tmpdir)
        assert sce.shape == (10, 4)
        assert "gene_symbols" in sce.row_data.column_names

        with open(os.path.join(tmpdir, "features.tsv"), "w") as f:
            f.write(
                "F1\tFeat1\nF2\tFeat2\nF3\tFeat3\nF4\tFeat4\nF5\tFeat5\nF6\tFeat6\nF7\tFeat7\nF8\tFeat8\nF9\tFeat9\nF10\tFeat10\n"
            )

        with pytest.warns(UserWarning, match="Both 'features.tsv' and 'genes.tsv' files are present"):
            sce = read_tenx_mtx(tmpdir)

        assert sce.shape == (10, 4)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmpfile:
        tmpname = tmpfile.name
    try:
        with h5py.File(tmpname, "w") as f:
            f.create_group("wrong_key")

        with pytest.raises(ValueError, match="is not a 10X V3 format"):
            read_tenx_h5(tmpname)

        with h5py.File(tmpname, "w") as f:
            mat_grp = f.create_group("matrix")
            mat_grp.create_dataset("shape", data=[10, 4])
            mat_grp.create_dataset("data", data=[1.0, 2.0])
            mat_grp.create_dataset("indices", data=[0, 1])
            mat_grp.create_dataset("indptr", data=[0, 1, 2, 2, 2])

            feat_grp = mat_grp.create_group("features")
            feat_grp.create_dataset("id", data=[b"F1", b"F2", b"F3", b"F4", b"F5", b"F6", b"F7", b"F8", b"F9", b"F10"])
            feat_grp.create_dataset("mismatched", data=[b"M1", b"M2", b"M3", b"M4", b"M5"])

            mat_grp.create_dataset("barcodes", data=[b"BC1", b"BC2", b"BC3", b"BC4"])

        with pytest.warns(UserWarning, match="These columns from h5 are ignored - mismatched"):
            sce = read_tenx_h5(tmpname)

        assert sce.shape == (10, 4)
        assert "id" in sce.row_data.column_names
        assert "mismatched" not in sce.row_data.column_names

    finally:
        if os.path.exists(tmpname):
            os.remove(tmpname)
