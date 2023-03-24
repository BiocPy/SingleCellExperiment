from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from summarizedexperiment.RangeSummarizedExperiment import RangeSummarizedExperiment
from genomicranges.GenomicRanges import GenomicRanges
from biocframe import BiocFrame

from typing import Union, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp
import pandas as pd
import anndata
from collections import OrderedDict

from mudata import MuData

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SingleCellExperiment(SummarizedExperiment):
    """Container class for single-cell experiments. Extends `SummarizedExperiment`
    to provide slots for embeddings and alternative experiments.
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
        typeCheckAlts: bool = True,
    ) -> None:
        """Initialize a single-cell experiment.

        This extends `SummarizedExperiment` and provides additional slots to
        store embeddings, alternative experiments, rowpairs and colpairs.

        Unlike R, numpy or scipy matrices do not have a notion of rownames
        and colnames. Hence, these matrices cannot be directly used as values either in
        assays or alternative experiments. Currently we strictily enforce type check in
        these cases. To relax these restrictions for alternative experiments, set 
        `typeCheckAlts` to `False`.

        One generally expects the same cells across all assays and alternative
         experiments.

        Note: Validation checks do not apply to rowpairs, colpairs.

        Args:
            assays (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): dictionary 
                of matrices, with assay names as keys and matrices represented as dense 
                (numpy) or sparse (scipy) matrices. All matrices across assays must 
                have the same dimensions (number of rows, number of columns).
            rowData (GenomicRanges, optional): features, must be the same length as 
                rows of the matrices in assays. Defaults to None.
            colData (Union[pd.DataFrame, BiocFrame], optional): sample data, must be 
                the same length as rows of the matrices in assays. Defaults to None.
            metadata (MutableMapping, optional): experiment metadata describing the 
                methods. Defaults to None.
            reducedDims (MutableMapping[str, Union[np.ndarray, sp.spmatrix]], optional): 
                lower dimensionality embeddings. Defaults to None.
            mainExperimentName (str, optional): main experiment name. Defaults to None.
            alter_exps (MutableMapping[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment], optional): 
                similar to assays, dictionary of alternative experiments with names as 
                keys. You would usually use this for multi-modal experiments performed 
                on the same sample (so all these experiments contain the same cells). Defaults to None.
            rowpairs (Union[np.ndarray, sp.spmatrix], optional): row 
                pairings/relationships between features. Defaults to None.
            colpairs (Union[np.ndarray, sp.spmatrix], optional): col 
                pairings/relationships between cells.  Defaults to None.
            typeCheckAlts (bool): strict type checking for alternative experiments. All 
                alternative experiments must be a derivative of `SummarizedExperiment`.
                Defaults to True.
        """
        super().__init__(
            assays=assays, rowData=rowData, colData=colData, metadata=metadata
        )

        self._validate_reducedDims(reducedDims)
        self._reducedDims = reducedDims

        self._mainExperimentName = mainExperimentName

        self._validate_altExpts(altExps, typeCheckAlts)
        self._typeCheckAlts = typeCheckAlts
        self._altExps = altExps

        self._validate_pairs(rowpairs)
        self._rowpairs = rowpairs

        self._validate_pairs(colpairs)
        self._colpairs = colpairs

    def _validate_reducedDims(
        self, reducedDims: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]
    ):
        """Validate reduced dimensions. all dimensions must contain embeddings for 
        all cells.

        Args:
            reducedDims (MutableMapping[str, Union[np.ndarray, sp.spmatrix]]): 
                embeddings to validate.

        Raises:
            TypeError: If embeddings are not a matrix (numpy, scipy) or pandas 
                Dataframe.
            TypeError: If reducedDims is not a dictionary like object.
            ValueError: length of dimensions do not match the number of cells.
        """
        if reducedDims is not None:

            if not isinstance(reducedDims, dict):
                raise TypeError("reducedDims is not a dictionary like object")

            for rdname, mat in reducedDims.items():
                if not (isinstance(mat, (np.ndarray, sp.spmatrix, pd.DataFrame))):
                    raise TypeError(
                        f"dimension: {rdname} must be either a numpy ndarray, scipy "
                        "matrix or a pandas DataFrame"
                    )

                if self._shape[1] != mat.shape[0]:
                    raise ValueError(
                        f"dimension: {rdname} does not contain embeddings for all cells"
                        f". should be {self._shape[1]}, but provided {mat.shape[0]}"
                    )

    def _validate_altExpts(
        self,
        altExps: Optional[
            MutableMapping[
                str,
                Union[
                    "SingleCellExperiment",
                    SummarizedExperiment,
                    RangeSummarizedExperiment,
                ],
            ]
        ],
        typeCheckAlts: bool,
    ):
        """Validate alternative experiments and optionally their types.

        Args:
            altExps (MutableMapping[str, Union[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment]]], optional): 
                similar to assays, dictionary of alternative experiments with names as 
                keys.
            typeCheckAlts (bool): strict type checking for alternative experiments.

        Raises:
            ValueError: if alternative experiments do not contain the same number of
                cells.
            TypeError: if alternative experiments is not a derivative of 
                `SummarizedExperiment`.
        """
        if altExps is not None:

            if not isinstance(altExps, dict):
                raise TypeError("altExps is not a dictionary like object")

            for altName, altExp in altExps.items():
                if self._shape[1] != altExp.shape[1]:
                    raise ValueError(
                        f"dimension: {altName} does not contain same number of cells."
                        f" should be {self._shape[1]}, but provided {altExp.shape[1]}"
                    )

                if typeCheckAlts is True:
                    if not isinstance(altExp, SummarizedExperiment):
                        raise TypeError(
                            f"alternative experiment {altName} is not a derivative of "
                            "`SummarizedExperiment`"
                        )

    def _validate_pairs(self, pairs: Optional[Union[np.ndarray, sp.spmatrix]]):
        """Validate row and column pairs.

        Currently only checks if they are dictionary like objects.

        Args:
            pairs (Optional[Union[np.ndarray, sp.spmatrix]]): pair to validate.

        Raises:
            TypeError: if pairs is not a dictionary like object.
        """
        if pairs is not None:
            if not isinstance(pairs, dict):
                raise TypeError("pair is not a dictionary like object")

    def _validate(self):
        """Internal method to validate the object.

        Raises:
            ValueError: when provided object does not contain columns of same length.
        """
        super()._validate()

        self._validate_reducedDims(self._reducedDims)
        self._validate_altExpts(self._altExps, self._typeCheckAlts)

    @property
    def reducedDims(
        self,
    ) -> Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]]:
        """Access dimensionality embeddings.

        Returns:
            Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]]: 
            all embeddings in the object. None if not available.
        """
        return self._reducedDims

    @reducedDims.setter
    def reducedDims(
        self,
        reducedDims: Optional[
            MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]
        ],
    ):
        """Set dimensionality embeddings.

        Args:
            reducedDims (MutableMapping[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]], optional): 
                new embeddings to set. Can be None.

        Raises:
            TypeError: reducedDims is not a dictionary.
        """
        self._validate_reducedDims(reducedDims)
        self._reducedDims = reducedDims

    @property
    def mainExperimentName(self) -> Optional[str]:
        """Access main experiment name.

        Returns:
            Optional[str]: name if available.
        """
        return self._mainExperimentName

    @mainExperimentName.setter
    def mainExperimentName(self, name: Optional[str]):
        """Set main experiment name.

        Args:
            name (Optional[str]): experiment name.
        """
        self._mainExperimentName = name

    @property
    def reducedDimNames(self) -> Optional[Sequence[str]]:
        """Access names of dimensionality embeddings.

        Returns:
            Optional[Sequence[str]]: all embeddings names.
        """

        if self._reducedDims is not None:
            return list(self._reducedDims.keys())

        return None

    def reducedDim(self, name: str) -> Union[np.ndarray, sp.spmatrix, pd.DataFrame]:
        """Access an embedding by name.

        Args:
            name (str): name of the embedding.

        Raises:
            ValueError: if embedding name does not exist.

        Returns:
            Union[np.ndarray, sp.spmatrix, pd.DataFrame]: access the underlying 
            numpy or scipy matrix.
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
        """Access alternative experiments.

        Returns:
            Optional[MutableMapping[str, Union[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment]]]: 
            alternative experiments.
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
        """Set alternative experiments.

        Returns:
            Optional[MutableMapping[str, Union[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment]]]: 
            alternative experiments.
        """
        self._validate_altExpts(altExps, self._typeCheckAlts)
        self._altExps = altExps

    def altExp(
        self, name: str
    ) -> Union[
        "SingleCellExperiment", SummarizedExperiment, RangeSummarizedExperiment,
    ]:
        """Access alternative experiment by name.

        Args:
            name (str): name of the alternative experiment.

        Raises:
            ValueError: name does not exist.

        Returns:
            Union[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment]:
             alternative experiment.
        """
        if name not in self._altExps:
            raise ValueError(f"Alt. Experiment {name} does not exist")

        return self._altExps[name]

    @property
    def rowPairs(self) -> Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]:
        """Access row pairings/relationships between features.

        Returns:
            Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: access rowpairs.
        """
        return self._rowpairs

    @rowPairs.setter
    def rowPairs(self, pairs: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]):
        """Set row pairs/relationships between features.

        Args:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: new row pairs to set.
        """
        self._validate_pairs(pairs)
        self._rowpairs = pairs

    @property
    def colPairs(self) -> Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]:
        """Access column pairs/relationships between cells.

        Returns:
            Optional[MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: access 
            colpairs.
        """
        return self._colpairs

    @colPairs.setter
    def colPairs(self, pairs: MutableMapping[str, Union[np.ndarray, sp.spmatrix]]):
        """Set column pairs/relationships between features.

        Args:
            MutableMapping[str, Union[np.ndarray, sp.spmatrix]]: new col pairs.
        """
        self._validate_pairs(pairs)
        self._colpairs = pairs

    def __str__(self) -> str:
        pattern = (
            f"Class SingleCellExperiment with {self.shape[0]} features and {self.shape[1]} cells \n"
            f"  mainExperimentName: {self._mainExperimentName if self._mainExperimentName is not None else None} \n"
            f"  assays: {list(self._assays.keys())} \n"
            f"  features: {self._rows.columns if self._rows is not None else None} \n"
            f"  cell annotations: {self._cols.columns if self._cols is not None else None} \n"
            f"  reduced dimensions: {self.reducedDimNames if self._reducedDims is not None else None} \n"
            f"  alternative experiments: {list(self._altExps.keys()) if self._altExps is not None else None}"
        )
        return pattern

    def _slice(
        self,
        args: Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]],
    ) -> Tuple[
        Union[pd.DataFrame, BiocFrame],
        Union[pd.DataFrame, BiocFrame],
        MutableMapping[str, Union[np.ndarray, sp.spmatrix]],
    ]:
        """Internal method to slice `SCE` by index.

        Args:
            args (Tuple[Union[Sequence[int], slice], Optional[Union[Sequence[int], slice]]]): 
            indices to slice. tuple can contains slices along dimensions.

        Raises:
            ValueError: Too many or few slices.

        Returns:
             Tuple[Union[pd.DataFrame, BiocFrame], Union[pd.DataFrame, BiocFrame], MutableMapping[str, Union[np.ndarray, sp.spmatrix]]]: 
             sliced row, cols and assays, embeddings and alternative experiments.
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
                    sliced_embeds = embeds.iloc[colIndices]
                else:
                    sliced_embeds = embeds[colIndices, :]

                new_reducedDims[rdname] = sliced_embeds

        if self._altExps is not None:
            new_altExps = OrderedDict()
            for ae in self._altExps.keys():
                new_altExps[ae] = self._altExps[ae][rowIndices, colIndices]

        return (new_rows, new_cols, new_assays, new_reducedDims, new_altExps)

    def __getitem__(self, args: tuple) -> "SingleCellExperiment":
        """Subset a `SingleCellExperiment`.
        
        Note: does not slice rowpairs and colpairs.

        Args:
            args (tuple): indices to slice. tuple can
                contains slices along dimensions.

        Raises:
            Exception: Too many ot few slices.

        Returns:
            SingleCellExperiment: new sliced `SingleCellExperiment` object.
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

    def toAnnData(
        self, alts: bool = False
    ) -> Union[anndata.AnnData, MutableMapping[str, anndata.AnnData]]:
        """Transform `SingleCellExperiment` object to `AnnData`.

        Args:
            alts (bool, optional): Also convert alternative experiments.

        Returns:
            Union[anndata.AnnData, MutableMapping[str, anndata.AnnData]]: 
            returns an AnnData representation, if alts is true, a dictionary 
            of anndata objects with their alternative experiment names.
        """

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()

        trows = self.rowData
        if isinstance(self.rowData, GenomicRanges):
            trows = self.rowData.toPandas()

        obj = anndata.AnnData(
            obs=self.colData,
            var=trows,
            uns=self.metadata,
            obsm=self.reducedDims,
            layers=layers,
            varp=self.rowPairs,
            obsp=self.colPairs,
        )

        if alts is True:
            adatas = None
            if self._altExps is not None:
                adatas = {}
                for altName, altExp in self._altExps.items():
                    adatas[altName] = altExp.toAnnData()

            return obj, adatas

        return obj

    def toMuData(self) -> MuData:
        """Transform `SingleCellExperiment` object to `MuData`.

        Returns:
            MuData: MuData representation.
        """
        mainData, altData = self.toAnnData(alts=True)

        expts = OrderedDict()
        mainName = self.mainExperimentName
        if self.mainExperimentName is None:
            mainName = "Unknown Modality"

        expts[mainName] = mainData

        if altData is not None:
            for exptName, expt in altData.items():
                expts[exptName] = expt

        return MuData(expts)
