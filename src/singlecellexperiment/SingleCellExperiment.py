from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from summarizedexperiment.RangeSummarizedExperiment import RangeSummarizedExperiment
from biocframe import BiocFrame

from typing import Union, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp
import pandas as pd
import anndata
from collections import OrderedDict

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SingleCellExperiment(SummarizedExperiment):
    """Container class for single-cell experiments. Extends `SummarizedExperiment`
    to provide slots for lower dimensionality embeddings and alternative experiments.
    """

    def __init__(
        self,
        assays: MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
        rowData: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        colData: Optional[Union[pd.DataFrame, BiocFrame]] = None,
        metadata: Optional[MutableMapping] = None,
        reducedDims: Optional[
            MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]
        ] = None,
        mainExperimentName: Optional[str] = None,
        altExps: Optional[
            MutableMapping[
                str,
                Union[
                    "SingleCellExperiment",
                    SummarizedExperiment,
                    RangeSummarizedExperiment,
                ],
            ]
        ] = None,
        rowpairs: Optional[Union[np.ndarray, sp.spmatrix]] = None,
        colpairs: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    ) -> None:
        """Initialize a single-cell experiment class.
            Note: Validation checks do not apply to rowpairs, colpairs.

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): dictionary of matrices,
                with assay names as keys and matrices represented as dense (numpy) or sparse (scipy) matrices.
                All matrices across assays must have the same dimensions (number of rows, number of columns).
            rowData (Union[pd.DataFrame, BiocFrame], optional): features, must be the same length as rows of the matrices in assays. Defaults to None.
            colData (Union[pd.DataFrame, BiocFrame], optional): cell data, must be the same length as the columns of the matrices in assays. Defaults to None.
            metadata (MutableMapping, optional): experiment metadata describing the methods. Defaults to None.
            reducedDims (MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]], optional): lower dimensionality embeddings. Defaults to None.
            mainExperimentName (str, optional): main experiment name. Defaults to None.
            alter_exps (MutableMapping[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment], optional): alternative experiments. Defaults to None.
            rowpairs (Union[np.ndarray, sp.spmatrix], optional): row pairings/relationships between features. Defaults to None.
            colpairs (Union[np.ndarray, sp.spmatrix], optional): col pairings/relationships between cells.  Defaults to None.
        """
        self._reducedDims = reducedDims
        self._mainExperimentName = mainExperimentName
        self._altExps = altExps
        self._rowpairs = rowpairs
        self._colpairs = colpairs

        super().__init__(
            assays=assays, rowData=rowData, colData=colData, metadata=metadata
        )

    def _validate(self):
        """Internal method to validate the object

        Raises:
            ValueError: when provided object does not contain columns of same length
        """
        super()._validate()

        base_dims = self._shape

        # check reducedDims
        if self._reducedDims is not None:
            for rdname, mat in self._reducedDims.items():
                if not (
                    isinstance(mat, np.ndarray)
                    or isinstance(mat, sp.spmatrix)
                    or isinstance(mat, pd.DataFrame)
                ):
                    raise TypeError(
                        f"dimension: {rdname} must be either a numpy ndarray, scipy matrix or a pandas DataFrame"
                    )

                if base_dims[1] != mat.shape[0]:
                    raise ValueError(
                        f"dimension: {rdname} does not contain embeddings for all cells. should be {base_dims[1]}, but provided {mat.shape[0]}"
                    )

    @property
    def reducedDims(
        self,
    ) -> MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]:
        """Access lower dimensionality embeddings

        Returns:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]: all embeddings in the object
        """
        return self._reducedDims

    @reducedDims.setter
    def reducedDims(
        self,
        reducedDims: MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]],
    ):
        """Set lower dimensionality embeddings

        Args:
            reducedDims (MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]): new embeddings

        Raises:
            TypeError: reducedDims is not a dictionary
        """
        if not isinstance(reducedDims, dict):
            raise TypeError("reducedDims is not a dictionary like object")

        self._reducedDims = reducedDims
        self._validate()

    @property
    def mainExperimentName(self) -> Optional[str]:
        """Access main experiment name

        Returns:
            Optional[str]: name if available
        """
        return self._mainExperimentName

    @mainExperimentName.setter
    def mainExperimentName(self, name: str):
        """Set main experiment name

        Args:
            name (str): experiment name
        """
        self._mainExperimentName = name

    @property
    def reducedDimNames(self) -> Optional[Sequence[str]]:
        """Access names of lower dimensionality embeddings

        Returns:
            Optional[Sequence[str]]: all embeddings names
        """

        if self._reducedDims is not None:
            return list(self._reducedDims.keys())

        return None

    def reducedDim(self, name: str) -> np.ndarray:
        """Access an embedding by name

        Args:
            name (str): name of the embedding

        Raises:
            ValueError: if embedding name does not exist

        Returns:
            np.ndarray: access the underlying numpy or scipy matrix
        """
        if name not in self._reducedDims:
            raise ValueError(f"Embedding: {name} does not exist")

        return self._reducedDims[name]

    @property
    def altExps(
        self,
    ) -> Optional[
        MutableMapping[
            str,
            Union[
                "SingleCellExperiment", SummarizedExperiment, RangeSummarizedExperiment,
            ],
        ]
    ]:
        """Access alternative experiments

        Returns:
            Optional[MutableMapping[str, Union["SingleCellExperiment", SummarizedExperiment, RangeSummarizedExperiment,]]]: alternative experiments
        """
        return self._altExps

    @altExps.setter
    def altExps(
        self,
        altExps: MutableMapping[
            str,
            Union[
                "SingleCellExperiment", SummarizedExperiment, RangeSummarizedExperiment,
            ],
        ],
    ):
        """Set alternative experiments

        Returns:
            Optional[MutableMapping[str, Union["SingleCellExperiment", SummarizedExperiment, RangeSummarizedExperiment,]]]: alternative experiments
        """
        if not isinstance(altExps, dict):
            raise TypeError("altExps is not a dictionary like object")

        self._altExps = altExps

    def altExp(
        self, name: str
    ) -> Union[
        "SingleCellExperiment", SummarizedExperiment, RangeSummarizedExperiment,
    ]:
        """Access alternative experiment by name

        Args:
            name (str): name of the alternative experiment

        Raises:
            ValueError: name does not exist

        Returns:
            Union["SingleCellExperiment", SummarizedExperiment, RangeSummarizedExperiment]: alternative experiment
        """
        if name not in self._altExps:
            raise ValueError(f"Alt. Experiment {name} does not exist")

        return self._altExps[name]

    @property
    def rowPairs(self) -> Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]:
        """Access row pairings/relationships between features.

        Returns:
            Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: access rowpairs
        """
        return self._rowpairs

    @rowPairs.setter
    def rowPairs(self, pairs: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]):
        """Set row pairings/relationships between features.

        Args:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: new row pairs
        """

        if not isinstance(pairs, dict):
            raise TypeError("rowpairs is not a dictionary like object")

        self._rowpairs = pairs

    @property
    def colPairs(self) -> Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]:
        """Access column pairings/relationships between cells.

        Returns:
            Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: access colpairs
        """
        return self._colpairs

    @colPairs.setter
    def colPairs(self, pairs: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]):
        """Set column pairings/relationships between features.

        Args:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: new col pairs
        """
        if not isinstance(pairs, dict):
            raise TypeError("colpairs is not a dictionary like object")

        self._colpairs = pairs

    def _slice(
        self,
        args: Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]],
    ) -> Tuple[
        Union[pd.DataFrame, BiocFrame],
        Union[pd.DataFrame, BiocFrame],
        MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
    ]:
        """Internal method to slice `SCE` by index

        Args:
            args (Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]]): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            ValueError: Too many slices

        Returns:
             Tuple[Union[pd.DataFrame, BiocFrame], Union[pd.DataFrame, BiocFrame], MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: sliced row, cols and assays.
        """

        if len(args) == 0:
            raise ValueError("Arguments must contain atleast one slice")

        rowIndices = args[0]
        colIndices = None

        if len(args) > 1:
            colIndices = args[1]
        elif len(args) > 2:
            raise ValueError("contains too many slices")

        new_rows = None
        new_cols = None
        new_assays = None
        new_reducedDims = None
        new_altExps = None

        if rowIndices is not None and self._rows is not None:
            if isinstance(self._rows, pd.DataFrame):
                new_rows = self._rows.iloc[rowIndices]
            else:
                new_rows = self._rows[rowIndices, :]

        if colIndices is not None and self._cols is not None:
            if isinstance(self._cols, pd.DataFrame):
                new_cols = self._cols.iloc[colIndices]
            else:
                new_cols = self._cols[colIndices, :]

        new_assays = self.subsetAssays(rowIndices=rowIndices, colIndices=colIndices)

        if self._reducedDims is not None:
            new_reducedDims = OrderedDict()
            for rdname, embeds in self._reducedDims.items():
                sliced_embeds = None
                if isinstance(embeds, pd.DataFrame):
                    sliced_embeds =  embeds.iloc[colIndices]
                else:
                    sliced_embeds = embeds[colIndices, :]

                new_reducedDims[rdname] = sliced_embeds

        if self._altExps is not None:
            new_altExps = OrderedDict()
            for ae in self._altExps.keys():
                new_altExps[ae] = self._altExps[ae][rowIndices, colIndices]

        return (new_rows, new_cols, new_assays, new_reducedDims, new_altExps)

    def __getitem__(self, args: tuple) -> "SingleCellExperiment":
        """Subset a SingleCellExperiment 
            Note: does not slice rowpairs and colpairs

        Args:
            args (tuple): indices to slice. tuple can
                contains slices along dimensions

        Raises:
            Exception: Too many slices

        Returns:
            SingleCellExperiment: new SingleCellExperiment object
        """
        new_rows, new_cols, new_assays, new_reducedDims, new_altExps = self._slice(args)

        return SingleCellExperiment(
            new_assays,
            new_rows,
            new_cols,
            self._metadata,
            new_reducedDims,
            self._mainExperimentName,
            new_altExps,
            self._rowpairs,
            self._colpairs,
        )

    def toAnnData(self) -> anndata.AnnData:
        """Transform `SingleCellExperiment` objects to `AnnData`, 
            Note: ignores alternative experiments.

        Returns:
            anndata.AnnData: returns an AnnData representation
        """

        # adatas = OrderedDict()

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()

        obj = anndata.AnnData(
            obs=self.colData,
            var=self.rowData,
            uns=self.metadata,
            obsm=self.reducedDims,
            layers=layers,
            varp=self.rowPairs,
            obsp=self.colPairs,
        )

        # name = "main_experiment"
        # if self.mainExperimentName is not None:
        #     name = self.mainExperimentName

        # adatas[name] = obj

        # if self._altExps is not None:
        #     for ae in self._altExps.keys():
        #         adatas[ae] = self._altExps[ae].toAnnData()

        return obj
