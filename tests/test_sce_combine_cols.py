import biocutils
import numpy as np
import pytest

from singlecellexperiment.SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_combine_columns(experiments):
    combined = biocutils.combine_columns(experiments.se_unnamed, experiments.se_unnamed_2)
    assert combined is not None
    assert isinstance(combined, SingleCellExperiment)
    assert len(combined.alternative_experiments) == 0
    assert len(combined.column_data["A"]) == 20

    combined2 = experiments.se_unnamed.combine_columns(experiments.se_unnamed_2)
    assert combined2 is not None
    assert isinstance(combined2, SingleCellExperiment)
    assert len(combined2.alternative_experiments) == 0
    assert len(combined2.column_data["A"]) == 20


def test_relaxed_combine_columns(experiments):
    ncols = 10
    nrows = 100
    test2 = experiments.se_unnamed_2.set_assays(
        {
            "counts": np.random.poisson(lam=10, size=(nrows, ncols)),
            "normalized": np.random.normal(size=(nrows, ncols)),
        },
        in_place=False,
    )

    with pytest.raises(Exception):
        combined = biocutils.combine_columns(experiments.se_unnamed, test2)

    combined = biocutils.relaxed_combine_columns(experiments.se_unnamed, test2)
    assert combined is not None
    assert isinstance(combined, SingleCellExperiment)
    assert len(combined.alternative_experiments) == 0
    assert len(combined.row_data["A"]) == 100

    combined2 = experiments.se_unnamed.relaxed_combine_columns(test2)
    assert combined2 is not None
    assert isinstance(combined2, SingleCellExperiment)
    assert len(combined2.alternative_experiments) == 0
    assert len(combined2.row_data["A"]) == 100


def test_combine_with_alts(experiments):
    combined = biocutils.combine_columns(experiments.se_with_alts1, experiments.se_with_alts2)
    assert combined is not None
    assert isinstance(combined, SingleCellExperiment)
    print(combined)
    assert len(combined.alternative_experiments) == 1
    assert len(combined.reduced_dim_names) == 1
    assert combined.reduced_dim_names == ["PCA"]
    assert len(combined.row_data["seqnames"]) == 3


def test_combine_utils_errors_and_masks():
    nrows_loc = 10
    ncols_loc = 4
    counts_loc = np.random.rand(nrows_loc, ncols_loc)

    sce1 = SingleCellExperiment(assays={"counts": counts_loc}, reduced_dimensions={"PCA": np.ones((ncols_loc, 2))})
    sce2 = SingleCellExperiment(assays={"counts": counts_loc}, reduced_dimensions={})

    combined = sce1.relaxed_combine_columns(sce2)
    pca = combined.get_reduced_dimension("PCA")

    assert isinstance(pca, np.ma.MaskedArray)
    assert np.all(pca.mask[ncols_loc:, :])
