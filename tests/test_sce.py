from random import random

import genomicranges
import numpy as np
import pandas as pd
import pytest
from biocframe import BiocFrame
from summarizedexperiment import SummarizedExperiment

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


def test_SCE_creation():
    tse = SingleCellExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    assert tse is not None
    assert isinstance(tse, sce)


def test_SCE_creation_with_alts():
    tse = SummarizedExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": tse},
    )

    assert tse is not None
    assert isinstance(tse, sce)


def test_SCE_creation_with_alts_should_fail():
    anrows = 200
    ancols = 2
    acounts = np.random.rand(anrows, ancols)
    adf_gr = pd.DataFrame({
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
    acol_data = pd.DataFrame({
        "treatment": ["ChIP", "Input"],
    })

    tse = SummarizedExperiment(assays={"counts": acounts}, row_data=adf_gr, column_data=acol_data)

    with pytest.raises(Exception):
        tse = SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=col_data,
            alternative_experiments={"alt": tse},
        )


def test_SCE_creation_modifications():
    rse = SummarizedExperiment(assays={"counts": counts}, row_data=row_data, column_data=col_data)

    tse = SingleCellExperiment(
        assays={"counts": counts},
        row_data=row_data,
        column_data=col_data,
        alternative_experiments={"alt": rse},
    )

    assert tse is not None
    assert isinstance(tse, sce)

    with pytest.raises(Exception):
        tse.set_reduced_dimension("something", np.random.rand(ncols - 1, 4), in_place=False)

    nassay_tse = tse.set_reduced_dimension("something", np.random.rand(tse.shape[1], 4), in_place=False)

    assert nassay_tse.get_reduced_dimension_names() != tse.get_reduced_dimension_names()

    tse.set_reduced_dimension("something", np.random.rand(tse.shape[1], 4), in_place=True)
    assert nassay_tse.get_reduced_dimension_names() == tse.get_reduced_dimension_names()


def test_SCE_different_alt_names():
    rse = SummarizedExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=pd.DataFrame(index=["ChIP"] * 6)
    )

    with pytest.raises(Exception):
        SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=col_data,
            alternative_experiments={"alt": rse},
        )

    with pytest.raises(Exception):
        SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=pd.DataFrame(index=["ChIP", "Input"] * 3),
            alternative_experiments={"alt": rse},
        )

    with pytest.raises(Exception):
        SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=pd.DataFrame(index=["ChIP", "Input", "Input"] * 2),
            alternative_experiments={"alt": rse},
        )


def test_SCE_dims():
    embeds = np.random.rand(counts.shape[1], 4)
    tse = SingleCellExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data, reduced_dimensions={"something": embeds}
    )

    assert tse is not None
    assert isinstance(tse, sce)
    assert tse.get_reduced_dimension_names() == ["something"]

    tse2 = SingleCellExperiment(
        assays={"counts": counts}, row_data=row_data, column_data=col_data, reduced_dims={"something": embeds}
    )

    assert tse2 is not None
    assert isinstance(tse2, sce)
    assert tse2.get_reduced_dimension_names() == ["something"]

    print(tse.get_reduced_dimension("something"), tse2.get_reduced_dimension("something"))

    assert np.allclose(tse.get_reduced_dimension("something"), tse2.get_reduced_dimension("something"))

    with pytest.raises(
        Exception, match="Either 'reduced_dims' or 'reduced_dimensions' should be provided, but not both."
    ):
        SingleCellExperiment(
            assays={"counts": counts},
            row_data=row_data,
            column_data=col_data,
            reduced_dims={"something": embeds},
            reduced_dimensions={"something": embeds},
        )


def test_validation_functions():
    from singlecellexperiment.SingleCellExperiment import (
        _validate_alternative_experiments,
        _validate_pairs,
        _validate_reduced_dims,
        _validate_size_factors,
    )

    shape = (nrows, ncols)

    with pytest.raises(ValueError, match="'reduced_dims' cannot be `None`"):
        _validate_reduced_dims(None, shape)
    with pytest.raises(TypeError, match="'reduced_dims' is not a dictionary"):
        _validate_reduced_dims("not_a_dict", shape)
    with pytest.raises(TypeError, match="must be a matrix-like object"):
        _validate_reduced_dims({"umap": "not_matrix"}, shape)
    with pytest.raises(ValueError, match="does not contain embeddings for all cells"):
        _validate_reduced_dims({"umap": np.zeros((ncols + 1, 2))}, shape)

    with pytest.raises(ValueError, match="'alternative_experiments' cannot be `None`"):
        _validate_alternative_experiments(None, shape, ["1", "2", "3", "4", "5", "6"])
    with pytest.raises(TypeError, match="'alternative_experiments' is not a dictionary"):
        _validate_alternative_experiments("not_a_dict", shape, ["1", "2", "3", "4", "5", "6"])
    with pytest.raises(TypeError, match="must be a 2-dimensional object"):
        _validate_alternative_experiments({"alt": "not_exp"}, shape, ["1", "2", "3", "4", "5", "6"])

    se_wrong_cells = SummarizedExperiment(assays={"counts": np.zeros((nrows, ncols + 1))})
    with pytest.raises(ValueError, match="does not contain same number of cells"):
        _validate_alternative_experiments({"alt": se_wrong_cells}, shape, ["1", "2", "3", "4", "5", "6"])

    se_wrong_names = SummarizedExperiment(
        assays={"counts": np.zeros((nrows, ncols))},
        column_data=BiocFrame({}, number_of_rows=ncols),
        column_names=["A", "B", "C", "D", "E", "F"],
    )
    with pytest.raises(Exception, match="Column names do not match"):
        _validate_alternative_experiments(
            {"alt": se_wrong_names}, shape, ["1", "2", "3", "4", "5", "6"], with_dim_names=True
        )

    with pytest.warns(UserWarning, match="Column names do not match"):
        _validate_alternative_experiments(
            {"alt": se_wrong_names}, shape, ["1", "2", "3", "4", "5", "6"], with_dim_names=False
        )

    with pytest.raises(TypeError, match="'size_factors' must be a sequence-like object"):
        _validate_size_factors(5, shape)
    with pytest.raises(ValueError, match="'size_factors' length must match the number of columns"):
        _validate_size_factors(np.zeros(ncols - 1), shape)

    with pytest.raises(TypeError, match="'row_pairs' is not a dictionary"):
        _validate_pairs("not_a_dict", nrows, "row_pairs")
    with pytest.raises(TypeError, match="must be a matrix-like object"):
        _validate_pairs({"p1": "not_matrix"}, nrows, "row_pairs")
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        _validate_pairs({"p1": np.zeros(nrows)}, nrows, "row_pairs")
    with pytest.raises(ValueError, match="must be a square matrix"):
        _validate_pairs({"p1": np.zeros((nrows, nrows + 1))}, nrows, "row_pairs")
