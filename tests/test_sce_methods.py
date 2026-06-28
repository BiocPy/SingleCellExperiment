from random import random

import genomicranges
import numpy as np
import pandas as pd
import pytest
from biocframe import BiocFrame
from summarizedexperiment import RangedSummarizedExperiment, SummarizedExperiment

from singlecellexperiment import SingleCellExperiment
from singlecellexperiment.SingleCellExperiment import SingleCellExperiment as sce

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
row_data = BiocFrame({
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
})

gr = genomicranges.GenomicRanges.from_pandas(row_data.to_pandas())

col_data = pd.DataFrame({
    "treatment": ["ChIP", "Input"] * 3,
})


def test_SCE_props():
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    assert tse is not None
    assert isinstance(tse, sce)

    assert tse.alternative_experiments == {}
    alt = SummarizedExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)
    tse.alternative_experiments = {"alt": alt}
    assert tse.alternative_experiments is not None

    assert tse.assays is not None
    assert tse.row_data is not None
    assert tse.col_data is not None

    assert tse.column_pairs == {}
    tse.column_pairs = {"random": np.random.rand(ncols, ncols)}
    assert tse.column_pairs is not None

    with pytest.raises(Exception):
        tse.row_pairs = counts

    with pytest.raises(Exception):
        tse.row_pairs = {"random": np.random.rand(nrows, 4)}

    assert tse.row_pairs == {}

    assert tse.main_experiment_name is None
    tse.main_experiment_name = "scrna-seq"
    assert tse.main_experiment_name is not None

    assert tse.reduced_dims == {}
    tse.reduced_dims = {"tsnooch": np.random.rand(ncols, 4)}
    with pytest.raises(Exception):
        tse.reduced_dims = {"tsnooch": np.random.rand(ncols - 1, 4)}
    assert tse.reduced_dims is not None

    assert tse.reduced_dim_names is not None
    assert len(tse.reduced_dim_names) == 1


def test_SCE_to_RSE():
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data, row_ranges=gr)

    rse = tse.to_rangedsummarizedexperiment()
    assert isinstance(rse, RangedSummarizedExperiment)
    assert not isinstance(rse, SingleCellExperiment)
    assert rse.shape == tse.shape
    assert rse.row_ranges is not None


def test_RSE_to_SCE():
    rse = RangedSummarizedExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data, row_ranges=gr)

    tse = SingleCellExperiment.from_rangedsummarizedexperiment(rse)
    assert isinstance(tse, SingleCellExperiment)
    assert tse.shape == rse.shape
    assert tse.row_ranges is not None


def test_SCE_to_SE():
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data, row_ranges=gr)

    se = tse.to_summarizedexperiment()
    assert isinstance(se, SummarizedExperiment)
    assert not isinstance(se, SingleCellExperiment)
    assert se.shape == tse.shape
    assert se.row_data is not None
    assert "seqnames" in se.row_data.column_names


def test_SE_to_SCE():
    se = SummarizedExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    tse = SingleCellExperiment.from_summarizedexperiment(se)
    assert isinstance(tse, SingleCellExperiment)
    assert tse.shape == se.shape


def test_size_factors():
    sf = np.random.rand(ncols)
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data, size_factors=sf)

    assert np.allclose(tse.size_factors, sf)
    assert np.allclose(tse.get_size_factors(), sf)
    assert "sizeFactors" in tse.column_data.column_names
    assert np.allclose(np.array(tse.column_data.column("sizeFactors")), sf)

    sf2 = np.random.rand(ncols)
    tse2 = tse.set_size_factors(sf2, in_place=False)
    assert np.allclose(tse2.size_factors, sf2)
    assert "sizeFactors" in tse2.column_data.column_names
    assert np.allclose(np.array(tse2.column_data.column("sizeFactors")), sf2)

    assert np.allclose(tse.size_factors, sf)  # original unchanged
    assert np.allclose(np.array(tse.column_data.column("sizeFactors")), sf)

    tse.set_size_factors(sf2, in_place=True)
    assert np.allclose(tse.size_factors, sf2)
    assert np.allclose(np.array(tse.column_data.column("sizeFactors")), sf2)

    tse_cleared = tse.set_size_factors(None, in_place=False)
    assert tse_cleared.size_factors is None
    assert "sizeFactors" not in tse_cleared.column_data.column_names

    tse_no_sf = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)
    assert tse_no_sf.size_factors is None
    assert tse_no_sf.get_size_factors(on_absence="none") is None
    assert "sizeFactors" not in tse_no_sf.column_data.column_names

    with pytest.warns(UserWarning):
        tse_no_sf.get_size_factors(on_absence="warn")

    with pytest.raises(ValueError):
        tse_no_sf.get_size_factors(on_absence="error")

    with pytest.raises(Exception):
        tse.set_size_factors(np.random.rand(ncols - 1))
    with pytest.raises(Exception):
        tse.set_size_factors(5)


def test_individual_pair_accessors():
    rp = np.random.rand(nrows, nrows)
    cp = np.random.rand(ncols, ncols)
    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        row_pairs={"rp1": rp},
        column_pairs={"cp1": cp},
    )

    assert np.allclose(tse.get_row_pair("rp1"), rp)
    assert np.allclose(tse.get_row_pair(0), rp)
    assert np.allclose(tse.get_column_pair("cp1"), cp)
    assert np.allclose(tse.get_column_pair(0), cp)

    rp2 = np.random.rand(nrows, nrows)
    tse2 = tse.set_row_pair("rp2", rp2, in_place=False)
    assert np.allclose(tse2.get_row_pair("rp2"), rp2)

    cp2 = np.random.rand(ncols, ncols)
    tse3 = tse.set_column_pair("cp2", cp2, in_place=False)
    assert np.allclose(tse3.get_column_pair("cp2"), cp2)

    with pytest.raises(IndexError):
        tse.get_row_pair(-1)
    with pytest.raises(IndexError):
        tse.get_row_pair(10)
    with pytest.raises(AttributeError):
        tse.get_row_pair("nonexistent")
    with pytest.raises(TypeError):
        tse.get_row_pair(3.5)


def test_copy_deepcopy():
    sf = np.random.rand(ncols)
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data, size_factors=sf)

    from copy import copy, deepcopy

    tse_copy = copy(tse)
    assert np.allclose(tse_copy.size_factors, sf)

    tse_deepcopy = deepcopy(tse)
    assert np.allclose(tse_deepcopy.size_factors, sf)


def test_alt_exp_workflows():
    rse = SummarizedExperiment(assays={"counts": np.random.rand(nrows, ncols)}, row_data=row_data, column_data=col_data)

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": rse},
        size_factors=np.random.rand(ncols),
    )

    swapped = tse.swap_alt_exp("alt", saved="main")
    assert isinstance(swapped, SingleCellExperiment)
    assert "main" in swapped.alternative_experiments
    assert "alt" not in swapped.alternative_experiments
    assert swapped.shape == (nrows, ncols)
    assert np.allclose(swapped.size_factors, tse.size_factors)

    f = ["groupA"] * (nrows // 2) + ["groupB"] * (nrows // 2)
    split_sce = tse.split_alt_exps(f, ref="groupA")
    assert "groupB" in split_sce.alternative_experiments
    assert split_sce.shape == (nrows // 2, ncols)
    assert split_sce.alternative_experiments["groupB"].shape == (nrows // 2, ncols)

    unsplit_sce = split_sce.unsplit_alt_exps(names=["groupB"])
    assert "groupB" not in unsplit_sce.alternative_experiments
    assert unsplit_sce.shape == (nrows, ncols)
