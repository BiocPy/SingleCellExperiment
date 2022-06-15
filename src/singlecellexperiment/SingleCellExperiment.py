from asyncore import close_all
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from summarizedexperiment.RangeSummarizedExperiment import (
    RangeSummarizedExperiment,
)
from genomicranges import GenomicRanges
from typing import Dict, List, Union, Any
import numpy as np
from scipy import sparse as sp
import pandas as pd
import logging
import anndata

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SingleCellExperiment(SummarizedExperiment):
    """Container class for single-cell experiments. Extends `SummarizedExperiment`
    to provide slots for lower dimensionality embeddings and alternative experiments.
    """

    def __init__(
        self,
        assays: Dict[str, Union[np.ndarray, sp.spmatrix]],
        rows: pd.DataFrame = None,
        cols: pd.DataFrame = None,
        metadata: Any = None,
        reducedDims: Dict[str, Union[np.ndarray, sp.spmatrix]] = None,
        main_exp_name: str = None,
        alter_exps: Dict[
            str,
            Union[
                "SingleCellExperiment",
                SummarizedExperiment,
                RangeSummarizedExperiment,
            ],
        ] = None,
        rowpairs: Union[np.ndarray, sp.spmatrix] = None,
        colpairs: Union[np.ndarray, sp.spmatrix] = None,
    ) -> None:
        """Initialize a single-cell experiment class

        Args:
            assays (Dict[str, Union[np.ndarray, sp.spmatrix]]): assays/experiment data
            rows (pd.DataFrame, optional): feature information. Defaults to None.
            cols (pd.DataFrame, optional): sample metadata. Defaults to None.
            metadata (Any, optional): experiment metadata. Defaults to None.
            reducedDims (Dict[str,Union[np.ndarray, sp.spmatrix]], optional): lower dimensionality embeddings. Defaults to None.
            main_exp_name (str, optional): main experiment name. Defaults to None.
            alter_exps (Union[&quot;SingleCellExperiment&quot;, SummarizedExperiment, RangeSummarizedExperiment], optional): alternative experiments. Defaults to None.
            rowpairs (Union[np.ndarray, sp.spmatrix], optional): row pairings/relationships between features. Defaults to None.
            colpairs (Union[np.ndarray, sp.spmatrix], optional): col pairings/relationships between cells.  Defaults to None.
        """
        super().__init__(assays, rows, cols, metadata)
        self._reducedDims = reducedDims
        self.main_exp_name = main_exp_name
        self.alter_exps = alter_exps
        self.rowpairs = rowpairs
        self.colpairs = colpairs

    def reducedDims(self) -> Dict[str, Union[np.ndarray, sp.spmatrix]]:
        """Access lower dimensionality embeddings

        Returns:
            Dict[str, Union[np.ndarray, sp.spmatrix]]: all embeddings in the object
        """
        return self._reducedDims

    def reducedDimNames(self) -> List[str]:
        """Access names of lower dimensionality embeddings

        Returns:
            List[str]: all names of embeddings
        """
        return list(self._reducedDims.keys())

    def reducedDim(self, name: str) -> Union[np.ndarray, sp.spmatrix]:
        """Access an embedding by name

        Args:
            name (str): name of the embedding

        Raises:
            Exception: if embedding name does not exist

        Returns:
            Union[np.ndarray, sp.spmatrix]: access the underlying numpy or scipy matrix
        """
        if name not in self._reducedDims:
            logging.error(f"Embedding {name} does not exist")
            raise Exception(f"Embedding {name} does not exist")

        return self._reducedDims[name]

    def altExps(
        self,
    ) -> Dict[
        str,
        Union[
            "SingleCellExperiment",
            SummarizedExperiment,
            RangeSummarizedExperiment,
        ],
    ]:
        """Access alternative experiments from SCE

        Returns:
            Dict[str, Union[
            "SingleCellExperiment",
            SummarizedExperiment,
            RangeSummarizedExperiment,
        ]]: alternative experiments
        """
        return self.alter_exps

    def altExp(
        self, name: str
    ) -> Union[
        "SingleCellExperiment",
        SummarizedExperiment,
        RangeSummarizedExperiment,
    ]:
        """Access alternative experiments from SCE

        Args:
            name (str): name of the alternative experiment

        Returns:
            Union[
            "SingleCellExperiment",
            SummarizedExperiment,
            RangeSummarizedExperiment,
        ]: alternative experiment by name
        """
        if name not in self.alter_exps:
            logging.error(f"Alt Experiment {name} does not exist")
            raise Exception(f"Alt Experiment {name} does not exist")

        return self.alter_exps[name]

    def rowPairs(self) -> Union[np.ndarray, sp.spmatrix]:
        """Access row pairings/relationships between features.

        Returns:
            Union[np.ndarray, sp.spmatrix]: access rowpairs
        """
        return self.rowpairs

    def colPairs(self) -> Union[np.ndarray, sp.spmatrix]:
        """Access col pairings/relationships between cells.

        Returns:
            Union[np.ndarray, sp.spmatrix]: access colpairs
        """
        return self.colpairs

    def __getitem__(self, args: tuple) -> "SingleCellExperiment":
        """Subset a SingleCellExperiment

        Args:
            args (tuple): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            Exception: Too many slices

        Returns:
            SingleCellExperiment: new SingleCellExperiment object containing the subset
        """
        rowIndices = args[0]
        colIndices = None

        if len(args) > 1:
            colIndices = args[1]
        elif len(args) > 2:
            logging.error(
                f"too many slices, args length must be 2 but provided {len(args)} "
            )
            raise Exception("contains too many slices")

        new_rows = None
        new_cols = None
        new_assays = None
        new_rowpairs = None
        new_colpairs = None
        new_reducedDims = None
        new_alt_exps = None

        if rowIndices is not None and self.rows is not None:
            if isinstance(self.rows, GenomicRanges):
                new_rows = self.rows[rowIndices]
            elif isinstance(self.rows, pd.DataFrame):
                new_rows = self.rows.iloc[rowIndices]

            if self.rowpairs is not None and (
                isinstance(self.rowpairs, np.ndarray)
                or isinstance(self.rowpairs, sp.spmatrix)
            ):
                new_rowpairs = self.rowpairs[rowIndices, rowIndices]

        if colIndices is not None and self.cols is not None:
            new_cols = self.cols.iloc[colIndices]

            if self.colpairs is not None and (
                isinstance(self.colpairs, np.ndarray)
                or isinstance(self.colpairs, sp.spmatrix)
            ):
                new_colpairs = self.colpairs[colIndices, colIndices]

        if self._reducedDims is not None:
            new_reducedDims = {}
            for rd in self._reducedDims.keys():
                new_reducedDims[rd] = self._reducedDims[rd][
                    colIndices,
                ]

        if self.alter_exps is not None:
            new_alt_exps = {}
            for ae in self.alter_exps.keys():
                new_alt_exps[ae] = self.alter_exps[ae][rowIndices, colIndices]

        new_assays = self.subsetAssays(
            rowIndices=rowIndices, colIndices=colIndices
        )

        return SingleCellExperiment(
            new_assays,
            new_rows,
            new_cols,
            self.metadata,
            new_reducedDims,
            self.main_exp_name,
            new_alt_exps,
            new_rowpairs,
            new_colpairs,
        )

    def toAnnData(self) -> anndata.AnnData:
        """Transform SCE to AnnData; currently ignores alternative experiments

        Returns:
            anndata.AnnData: return an AnnData representation
        """

        return anndata.AnnData(
            obs=self.cols,
            var=self.rows,
            uns=self.metadata,
            obsm=self._reducedDims,
            layers=self._assays,
            varp=self.rowpairs,
            obsp=self.colpairs,
        )
