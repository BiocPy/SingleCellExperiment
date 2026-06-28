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


def test_sce_alias_and_deprecated_paths():
    with pytest.warns(DeprecationWarning, match="'reduced_dims' is deprecated"):
        sce_dep = SingleCellExperiment(assays={"counts": counts}, reduced_dims={"PCA": np.zeros((ncols, 2))})
        assert "PCA" in sce_dep.reduced_dim_names

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        reduced_dimensions={"PCA": np.zeros((ncols, 2)), "UMAP": np.zeros((ncols, 3))},
    )

    assert "PCA" in tse.get_reduced_dims()

    sce2 = tse.set_reduced_dims({"TSNE": np.zeros((ncols, 2))}, in_place=False)
    assert "TSNE" in sce2.get_reduced_dims()
    assert "PCA" in tse.get_reduced_dims()

    with pytest.warns(UserWarning, match="use 'set_reduced_dimensions' instead"):
        tse.reduced_dims = {"TSNE": np.zeros((ncols, 2))}

    assert "TSNE" in tse.reduced_dim_names

    with pytest.warns(UserWarning, match="use 'set_reduced_dimensions' instead"):
        tse.reduced_dimensions = {"PCA": np.zeros((ncols, 2))}

    assert "PCA" in tse.reduced_dim_names

    with pytest.warns(UserWarning, match="use 'set_reduced_dimension_names' instead"):
        tse.reduced_dim_names = ["PCA_new"]

    assert "PCA_new" in tse.reduced_dim_names

    with pytest.warns(UserWarning, match="use 'set_reduced_dimension_names' instead"):
        tse.reduced_dimension_names = ["PCA_brand_new"]

    assert "PCA_brand_new" in tse.reduced_dimension_names

    tse = tse.set_reduced_dim_names(["PCA"], in_place=False)
    with pytest.raises(ValueError, match="Length of 'names' does not match"):
        tse.set_reduced_dim_names(["A", "B"])

    sce_renamed = tse.set_reduced_dim_names(["PCA_renamed"], in_place=False)
    assert "PCA_renamed" in sce_renamed.reduced_dim_names

    with pytest.raises(IndexError, match="Index cannot be negative"):
        tse.get_reduced_dimension(-1)

    with pytest.raises(IndexError, match="Index greater than the number of reduced dimensions"):
        tse.get_reduced_dimension(10)

    assert tse.get_reduced_dimension(0).shape == (ncols, 2)

    with pytest.raises(AttributeError, match="does not exist"):
        tse.get_reduced_dimension("TSNE")

    assert tse.get_reduced_dimension("PCA").shape == (ncols, 2)
    assert tse.reduced_dim("PCA").shape == (ncols, 2)
    assert tse.reduced_dimension("PCA").shape == (ncols, 2)

    with pytest.raises(TypeError, match="must be a string or integer"):
        tse.get_reduced_dimension([1, 2])

    with pytest.raises(ValueError, match="Length of 'names' does not match"):
        tse.set_alternative_experiment_names(["alt1", "alt2"])

    se1 = SummarizedExperiment(assays={"counts": counts})
    sce_alt = SingleCellExperiment(assays={"counts": counts}, alternative_experiments={"alt1": se1})
    with pytest.warns(UserWarning, match="use 'set_alternative_experiment_names' instead"):
        sce_alt.alternative_experiment_names = ["alt1_new"]

    assert "alt1_new" in sce_alt.alternative_experiment_names

    with pytest.raises(IndexError, match="Index cannot be negative"):
        sce_alt.get_alternative_experiment(-1)

    with pytest.raises(IndexError, match="Index greater than the number of alternative experiments"):
        sce_alt.get_alternative_experiment(5)

    with pytest.raises(AttributeError, match="does not exist"):
        sce_alt.get_alternative_experiment("nonexistent")

    with pytest.raises(TypeError, match="must be a string or integer"):
        sce_alt.get_alternative_experiment([1])

    assert isinstance(sce_alt.alternative_experiment(0), SummarizedExperiment)

    with pytest.warns(UserWarning, match="use 'set_row_pairs' instead"):
        tse.row_pairs = {"rp": np.zeros((nrows, nrows))}

    with pytest.warns(UserWarning, match="use 'set_column_pairs' instead"):
        tse.column_pairs = {"cp": np.zeros((ncols, ncols))}

    with pytest.warns(UserWarning, match="use 'set_row_pair_names' instead"):
        tse.row_pair_names = ["rp_new"]

    with pytest.warns(UserWarning, match="use 'set_column_pair_names' instead"):
        tse.column_pair_names = ["cp_new"]

    with pytest.raises(IndexError, match="Index cannot be negative"):
        tse.get_row_pair(-1)

    with pytest.raises(IndexError, match="Index greater than the number of row pairs"):
        tse.get_row_pair(5)

    with pytest.raises(AttributeError, match="does not exist"):
        tse.get_row_pair("nonexistent")

    with pytest.raises(TypeError, match="must be a string or integer"):
        tse.get_row_pair([1])

    with pytest.raises(IndexError, match="Index cannot be negative"):
        tse.get_column_pair(-1)

    with pytest.raises(IndexError, match="Index greater than the number of column pairs"):
        tse.get_column_pair(5)

    with pytest.raises(AttributeError, match="does not exist"):
        tse.get_column_pair("nonexistent")

    with pytest.raises(TypeError, match="must be a string or integer"):
        tse.get_column_pair([1])


def test_sce_workflow_corner_cases():
    se = SummarizedExperiment(assays={"counts": counts}, column_data=col_data)
    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": se},
        reduced_dimensions={"PCA": np.zeros((ncols, 2))},
    )

    swapped = tse.swap_alt_exp("alt", with_col_data=False)
    assert swapped.column_data.shape == se.column_data.shape
    assert len(swapped.reduced_dim_names) == 0

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in row_data"):
        tse.split_alt_exps("nonexistent")
    with pytest.raises(ValueError, match="Length of 'f' must match the number of rows"):
        tse.split_alt_exps([1, 2])
    with pytest.raises(ValueError, match="Reference group 'wrong' not found in groups"):
        tse.split_alt_exps(["G1", "G2"] * int(nrows / 2), ref="wrong")

    sce_split = tse.copy()
    sce_split.split_alt_exps(["G1", "G2"] * int(nrows / 2), ref="G1", in_place=True)
    assert "G2" in sce_split.alternative_experiment_names

    assert isinstance(tse.unsplit_alt_exps(names=[]), SingleCellExperiment)
    sce_unsplit = tse.copy()
    assert sce_unsplit.unsplit_alt_exps(names=[], in_place=True) is sce_unsplit

    with pytest.raises(ValueError, match="not found"):
        tse.unsplit_alt_exps(names=["wrong"])

    se_alt = SummarizedExperiment(assays={"counts": np.random.rand(5, ncols)}, column_data=col_data)
    rse_alt = RangedSummarizedExperiment(assays={"counts": np.random.rand(5, ncols)}, column_data=col_data)

    sce_multi = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"se_alt": se_alt, "rse_alt": rse_alt},
    )

    unsplit = sce_multi.unsplit_alt_exps()
    assert unsplit.shape[0] == nrows + 5 + 5

    sce_multi_ip = sce_multi.copy()
    sce_multi_ip.unsplit_alt_exps(in_place=True)
    assert sce_multi_ip.shape[0] == nrows + 5 + 5
