from typing import Union, Dict, Any
from genomicranges import GenomicRanges
import numpy as np
from scipy import sparse as sp
import pandas as pd
import anndata

from .SingleCellExperiment import SingleCellExperiment as sce
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from summarizedexperiment.RangeSummarizedExperiment import (
    RangeSummarizedExperiment,
)

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def SingleCellExperiment(
    assays: Dict[str, Union[np.ndarray, sp.spmatrix]],
    rowData: pd.DataFrame = None,
    colData: pd.DataFrame = None,
    metadata: Any = None,
    reducedDims: Dict[str, Union[np.ndarray, sp.spmatrix]] = None,
    mainExpName: str = None,
    alterExps: Dict[
        str,
        Union[
            sce,
            SummarizedExperiment,
            RangeSummarizedExperiment,
        ],
    ] = None,
    rowPairs: Union[np.ndarray, sp.spmatrix] = None,
    colPairs: Union[np.ndarray, sp.spmatrix] = None,
) -> sce:
    """Validates and creates a `SingleCellExperiment` object

    Args:
        assays (Dict[str, Union[np.ndarray, sp.spmatrix]]): assays/experiment data
        rowData (pd.DataFrame, optional): feature information. Defaults to None.
        colData (pd.DataFrame, optional): sample metadata. Defaults to None.
        metadata (Any, optional): experiment metadata. Defaults to None.
        reducedDims (Dict[str,Union[np.ndarray, sp.spmatrix]], optional): lower dimensionality embeddings. Defaults to None.
        mainExpName (str, optional): main experiment name. Defaults to None.
        alterExps (Dict[str, Union[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment, ], ], optional): alternative experiments. Defaults to None.
        rowPairs (Union[np.ndarray, sp.spmatrix], optional): row pairings/relationships between features. Defaults to None.
        colPairs (Union[np.ndarray, sp.spmatrix], optional): col pairings/relationships between cells.  Defaults to None.

    Raises:
        Exception: Assays must contain atleast one experiment matrix
        Exception: Matrix dimensions not consistent
        Exception: Matrix dimensions does not match rowData
        Exception: Matrix dimensions does not match colData

    Returns:
        SingleCellExperiment: data represented as SCE
    """
    if (
        assays is None
        or (not isinstance(assays, dict))
        or len(assays.keys()) == 0
    ):
        raise Exception(
            f"{assays} must be a dictionary and contain atleast a single numpy/scipy matrix"
        )

    row_lengths = None
    if rowData is not None and isinstance(rowData, GenomicRanges):
        row_lengths = len(rowData)
    elif rowData is not None:
        row_lengths = rowData.shape[0]

    if colData is not None:
        col_lengths = colData.shape[0]

    # make sure all matrices are the same shape

    matrix_lengths = None
    for d in assays:
        if matrix_lengths is None:
            matrix_lengths = assays[d].shape

        if matrix_lengths != assays[d].shape:
            raise Exception(
                f"matrix dimensions don't match across assays: {d}"
            )

    # are rows same length ?
    if rowData is not None and row_lengths != matrix_lengths[0]:
        raise Exception(
            f"matrix dimensions does not match rowData/rowRanges: {row_lengths} :: {matrix_lengths[0]}"
        )

    # are cols same length ?
    if colData is not None and col_lengths != matrix_lengths[1]:
        raise Exception(
            f"matrix dimensions does not match rowData/rowRanges: {col_lengths} :: {matrix_lengths[1]}"
        )

    # are rowpairs same length ?
    if rowPairs is not None and rowPairs.shape[0] != matrix_lengths[0]:
        raise Exception(
            f"matrix dimensions does not match rowPairs: {rowPairs.shape[0]} :: {matrix_lengths[0]}"
        )

    # are colpairs same length ?
    if colPairs is not None and colPairs.shape[1] != matrix_lengths[1]:
        raise Exception(
            f"matrix dimensions does not match rowPairs: {colPairs.shape[0]} :: {matrix_lengths[1]}"
        )

    if mainExpName is None:
        mainExpName = "unnamed experiment"

    return sce(
        assays=assays,
        rows=rowData,
        cols=colData,
        metadata=metadata,
        reducedDims=reducedDims,
        main_exp_name=mainExpName,
        alter_exps=alterExps,
        rowpairs=rowPairs,
        colpairs=colPairs,
    )


def fromAnnData(adata: anndata.AnnData) -> sce:
    """Represent AnnData object as SingleCellExperiment

    Args:
        adata (AnnData): AnnData object

    Returns:
        SingleCellExperiment: single-cell experiment representation of AnnData
    """

    return sce(
        assays=adata.layers,
        rows=adata.var,
        cols=adata.obs,
        metadata=adata.uns,
        reducedDims=adata.obsm,
        rowpairs=adata.varp,
        colpairs=adata.obsp,
    )


def fromH5AD(path: str) -> sce:
    """Represent AnnData object as SingleCellExperiment

    Args:
        path (str): path to a H5AD file

    Returns:
        SingleCellExperiment: single-cell experiment representation of AnnData
    """

    adata = anndata.read_h5ad(path)
    return fromAnnData(adata)
