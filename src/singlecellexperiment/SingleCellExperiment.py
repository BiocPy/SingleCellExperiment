from collections import OrderedDict
from typing import MutableMapping, Optional, Sequence, Union

import anndata
import numpy as np
import pandas as pd
from mudata import MuData
from scipy import sparse as sp
from genomicranges import GenomicRanges
from summarizedexperiment.BaseSE import BaseSE
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from summarizedexperiment.types import BiocOrPandasFrame, MatrixTypes, SlicerArgTypes

from ._types import MatrixTypesWithFrame

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SingleCellExperiment(SummarizedExperiment):
    """Container class for single-cell experiments. Extends `SummarizedExperiment`
    to provide slots for embeddings and alternative experiments.
    """

    def __init__(
        self,
        assays: MutableMapping[str, MatrixTypes],
        rowData: Optional[BiocOrPandasFrame] = None,
        colData: Optional[BiocOrPandasFrame] = None,
        metadata: Optional[MutableMapping] = None,
        reducedDims: Optional[MutableMapping[str, MatrixTypesWithFrame]] = None,
        mainExperimentName: Optional[str] = None,
        altExps: Optional[
            MutableMapping[
                str,
                BaseSE,
            ]
        ] = None,
        rowpairs: Optional[MatrixTypesWithFrame] = None,
        colpairs: Optional[MatrixTypesWithFrame] = None,
        typeCheckAlts: bool = True,
    ) -> None:
        """Initialize a single-cell experiment.

        This extends `SummarizedExperiment` and provides additional slots to
        store embeddings, alternative experiments, rowpairs and colpairs.

        Unlike R, numpy or scipy matrices do not have a notion of rownames
        and colnames. Hence, these matrices cannot be directly used as values either in
        assays or alternative experiments. Currently we strictly enforce type check in
        these cases. To relax these restrictions for alternative experiments, set
        `typeCheckAlts` to `False`.

        If you are using alternative experiment slot, the number of cells should match
        with the parent. If not the cells are not shared so they don't belong in alternative experiments!

        Note: Validation checks do not apply to rowpairs, colpairs.

        Args:
            assays (MutableMapping[str, MatrixTypes]): dictionary
                of matrices, with assay names as keys and matrices represented as dense
                (numpy) or sparse (scipy) matrices. All matrices across assays must
                have the same dimensions (number of rows, number of columns).
            rowData (BiocOrPandasFrame, optional): features, must be the same length as
                rows of the matrices in assays. Defaults to None.
            colData (BiocOrPandasFrame, optional): sample data, must be
                the same length as rows of the matrices in assays. Defaults to None.
            metadata (MutableMapping, optional): experiment metadata describing the
                methods. Defaults to None.
            reducedDims (MutableMapping[str, MatrixTypesWithFrame], optional):
                lower dimensionality embeddings. Defaults to None.
            mainExperimentName (str, optional): main experiment name. Defaults to None.
            alter_exps (MutableMapping[SingleCellExperiment, SummarizedExperiment, RangeSummarizedExperiment], optional):
                similar to assays, dictionary of alternative experiments with names as
                keys. You would usually use this for multi-modal experiments performed
                on the same sample (so all these experiments contain the same cells). Defaults to None.
            rowpairs (MatrixTypesWithFrame, optional): row
                pairings/relationships between features. Defaults to None.
            colpairs (MatrixTypesWithFrame, optional): col
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
        self, reducedDims: MutableMapping[str, MatrixTypesWithFrame]
    ):
        """Validate reduced dimensions. all dimensions must contain embeddings for
        all cells.

        Args:
            reducedDims (MutableMapping[str, MatrixTypesWithFrame]):
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
                BaseSE,
            ]
        ],
        typeCheckAlts: bool,
    ):
        """Validate alternative experiments and optionally their types.

        Args:
            altExps (MutableMapping[str, BaseSE], optional):
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
                    if not isinstance(altExp, BaseSE):
                        raise TypeError(
                            f"alternative experiment {altName} is not a derivative of "
                            "`SummarizedExperiment`"
                        )

    def _validate_pairs(self, pairs: Optional[MatrixTypesWithFrame]):
        """Validate row and column pairs.

        Currently only checks if they are dictionary like objects.

        Args:
            pairs (Optional[MatrixTypesWithFrame]): pair to validate.

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
    ) -> Optional[MutableMapping[str, MatrixTypesWithFrame]]:
        """Access dimensionality embeddings.

        Returns:
            (MutableMapping[str, MatrixTypesWithFrame], optional):
            all embeddings in the object. None if not available.
        """
        return self._reducedDims

    @reducedDims.setter
    def reducedDims(
        self,
        reducedDims: Optional[MutableMapping[str, MatrixTypesWithFrame]],
    ):
        """Set dimensionality embeddings.

        Args:
            reducedDims (MutableMapping[str, MatrixTypesWithFrame], optional):
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
            (str, optional): name if available.
        """
        return self._mainExperimentName

    @mainExperimentName.setter
    def mainExperimentName(self, name: Optional[str]):
        """Set main experiment name.

        Args:
            name (str, optional): experiment name to set.
        """
        self._mainExperimentName = name

    @property
    def reducedDimNames(self) -> Optional[Sequence[str]]:
        """Access names of dimensionality embeddings.

        Returns:
            (Sequence[str], optional): all embeddings names if available.
        """

        if self._reducedDims is not None:
            return list(self._reducedDims.keys())

        return None

    def reducedDim(self, name: str) -> MatrixTypesWithFrame:
        """Access an embedding by name.

        Args:
            name (str): name of the embedding.

        Raises:
            ValueError: if embedding name does not exist.

        Returns:
            MatrixTypesWithFrame: access the underlying
            numpy or scipy matrix.
        """
        if name not in self._reducedDims:
            raise ValueError(f"Embedding: {name} does not exist")

        return self._reducedDims[name]

    @property
    def altExps(
        self,
    ) -> Optional[MutableMapping[str, BaseSE,]]:
        """Access alternative experiments.

        Returns:
            (MutableMapping[str, BaseSE], optional):
            alternative experiments if available.
        """
        return self._altExps

    @altExps.setter
    def altExps(
        self,
        altExps: MutableMapping[str, BaseSE],
    ):
        """Set alternative experiments.

        Returns:
            Optional[MutableMapping[str, BaseSE]:
            alternative experiments.
        """
        self._validate_altExpts(altExps, self._typeCheckAlts)
        self._altExps = altExps

    def altExp(self, name: str) -> BaseSE:
        """Access alternative experiment by name.

        Args:
            name (str): name of the alternative experiment.

        Raises:
            ValueError: name does not exist.

        Returns:
            BaseSE: alternative experiment.
        """
        if name not in self._altExps:
            raise ValueError(f"Alt. Experiment {name} does not exist")

        return self._altExps[name]

    @property
    def rowPairs(self) -> Optional[MutableMapping[str, MatrixTypesWithFrame]]:
        """Access row pairings/relationships between features.

        Returns:
            Optional[MutableMapping[str, MatrixTypesWithFrame]]: access rowpairs.
        """
        return self._rowpairs

    @rowPairs.setter
    def rowPairs(self, pairs: MutableMapping[str, MatrixTypesWithFrame]):
        """Set row pairs/relationships between features.

        Args:
            MutableMapping[str, MatrixTypesWithFrame]: new row pairs to set.
        """
        self._validate_pairs(pairs)
        self._rowpairs = pairs

    @property
    def colPairs(self) -> Optional[MutableMapping[str, MatrixTypesWithFrame]]:
        """Access column pairs/relationships between cells.

        Returns:
            Optional[MutableMapping[str, MatrixTypesWithFrame]]: access
            colpairs.
        """
        return self._colpairs

    @colPairs.setter
    def colPairs(self, pairs: MutableMapping[str, MatrixTypesWithFrame]):
        """Set column pairs/relationships between features.

        Args:
            MutableMapping[str, MatrixTypesWithFrame]: new col pairs.
        """
        self._validate_pairs(pairs)
        self._colpairs = pairs

    def __str__(self) -> str:
        pattern = (
            f"Class SingleCellExperiment with {self.shape[0]} features and {self.shape[1]} cells \n"
            f"  mainExperimentName: {self.mainExperimentName if self.mainExperimentName is not None else None} \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  features: {self.rowData.columns if self.rowData is not None else None} \n"
            f"  cell annotations: {self.colData.columns if self.colData is not None else None} \n"
            f"  reduced dimensions: {self.reducedDimNames if self.reducedDims is not None else None} \n"
            f"  alternative experiments: {list(self.altExps.keys()) if self.altExps is not None else None}"
        )
        return pattern

    def __getitem__(self, args: SlicerArgTypes) -> "SingleCellExperiment":
        """Subset a `SingleCellExperiment`.

        Note: does not slice rowpairs and colpairs.

        Args:
            args (SlicerArgTypes): indices to slice. tuple can
                contains slices along dimensions.

        Raises:
            Exception: Too many or too few slices.

        Returns:
            SingleCellExperiment: new sliced `SingleCellExperiment` object.
        """
        sliced_objs = self._slice(args)

        new_reducedDims = None
        if self.reducedDims is not None:
            new_reducedDims = OrderedDict()
            for rdname, embeds in self.reducedDims.items():
                sliced_embeds = None
                if isinstance(embeds, pd.DataFrame):
                    sliced_embeds = embeds.iloc[sliced_objs.colIndices]
                else:
                    sliced_embeds = embeds[sliced_objs.colIndices, :]

                new_reducedDims[rdname] = sliced_embeds

        new_altExps = None
        if self._altExps is not None:
            new_altExps = OrderedDict()
            for ae in self._altExps.keys():
                new_altExps[ae] = self._altExps[ae][
                    sliced_objs.rowIndices, sliced_objs.colIndices
                ]

        return SingleCellExperiment(
            sliced_objs.assays,
            sliced_objs.rowData,
            sliced_objs.colData,
            self.metadata,
            new_reducedDims,
            self.mainExperimentName,
            new_altExps,
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
            if self.altExps is not None:
                adatas = {}
                for altName, altExp in self.altExps.items():
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
