from typing import Optional, Sequence, MutableMapping
from functools import reduce
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..SingleCellExperiment import SingleCellExperiment


def concat(
    sces: Sequence[SingleCellExperiment],
    how: str = "outer",
    mainExperimentName: Optional[str] = None,
    metadata: Optional[MutableMapping] = None,
) -> SingleCellExperiment:
    """Concatenate SingleCellExperiment objects.
    Assume we are working with RNA-Seq experiments for now.

    Parameters
    ----------
    sces: Sequence[SingleCellExperiment]
        A sequence of SingleCellExperiment objects.
    how: str
        Method to join SingleCellExperiment objects.
    mainExperimentName: str
        Name of the new SingleCellExperiment.
    metadata: str
        Metadata of the new SingleCellExperiment.
    """

    # we assume that sces have the same assay structure (same number of rows and cols)
    new_assays = {}
    assay_names = sces[0].assays.keys()
    for assay_name in assay_names:
        curr_assays = []
        for sce in sces:
            curr_assay = sce.assays[assay_name]
            curr_assays.append(
                pd.DataFrame(
                    curr_assay, columns=sce.colData.index, index=sce.rowData.index
                )
                if isinstance(curr_assay, np.ndarray)
                else pd.DataFrame.sparse.from_spmatrix(
                    curr_assay, columns=sce.colData.index, index=sce.rowData.index
                )
            )

        merged_assays = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how=how
            ),
            curr_assays,
        )
        merged_assays = merged_assays.sort_index().sort_values(
            by=merged_assays.columns.tolist()
        )
        merged_assays = sp.csr_matrix(merged_assays.values)
        new_assays[assay_name] = merged_assays

    rowDatas = []
    colDatas = []
    for sce in sces:
        rowDatas.append(sce.rowData)
        colDatas.append(sce.colData)

    new_rowData = reduce(
        lambda left, right: left.combine_first(right), rowDatas
    ).sort_index()
    new_colData = reduce(
        lambda left, right: left.combine_first(right), colDatas
    ).sort_index()

    return SingleCellExperiment(
        assays=new_assays,
        rowData=new_rowData,
        colData=new_colData,
        metadata=metadata,
        reducedDims=None,
        mainExperimentName=mainExperimentName,
        altExps=None,
        rowpairs=None,
        colpairs=None,
    )
