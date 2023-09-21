from collections import OrderedDict
from typing import Dict, List, Optional, Union

from genomicranges import GenomicRanges, GenomicRangesList
from pandas import DataFrame
from summarizedexperiment.RangedSummarizedExperiment import (
    GRangesOrGRangesList,
    RangedSummarizedExperiment,
)
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from summarizedexperiment.types import BiocOrPandasFrame, MatrixTypes, SlicerArgTypes

from ._types import MatrixTypesWithFrame

try:
    from anndata import AnnData
except ImportError:
    pass

try:
    from mudata import MuData
except ImportError:
    pass

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


class SingleCellExperiment(RangedSummarizedExperiment):
    """Container class for single-cell experiments, extending
    :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment` to provide slots for
    embeddings and alternative experiments that share the same cells.

    In contrast to R, :py:class:`~numpy.ndarray` or scipy matrices are unnamed and do not contain rownames and colnames.
    Consequently, these matrices cannot be directly used as values in assays or alternative experiments.
    We strictly enforce type checks in these cases.

    To relax these restrictions for alternative experiments, set
    :py:attr:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment.type_check_alternative_experiments` to
    False.

    If you are using the alternative experiment slot, the number of cells must match the parent experiment.
    Otherwise, these cells do not share the same sample or annotations and cannot be set in alternative experiments!

    Note: Validation checks do not apply to ``row_pairs`` or ``col_pairs``.

    Attributes:
        assays (Dict[str, MatrixTypes]): Dictionary of matrices with assay names as keys, and 2-dimensional matrices
            represented as :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

            Alternatively, you may use any 2-dimensional matrix that contains the property ``shape``
            and implements the slice operation using the ``__getitem__`` dunder method.

            All matrices in ``assays`` must be 2-dimensional and have the same shape
            (number of rows, number of columns).

        row_ranges (GRangesOrGRangesList, optional): Genomic features, must be the same length as rows of the matrices
            in assays.

        row_data (BiocOrPandasFrame, optional): Features, must be the same length as rows of the matrices in assays.
            Features may be either a :py:class:`~pandas.DataFrame` or
            :py:class:`~biocframe.BiocFrame.BiocFrame`.

        col_data (BiocOrPandasFrame, optional): Sample data, must be the same length as columns of the matrices in
            assays. Sample Information may be either a :py:class:`~pandas.DataFrame` or
            :py:class:`~biocframe.BiocFrame.BiocFrame`.

        metadata (Dict, optional): Additional experimental metadata describing the methods.

        reduced_dims (Dict[str, MatrixTypesWithFrame], optional): Slot for lower dimensionality embeddings.

            Usually a dictionary with the embedding method as keys (e.g., t-SNE, UMAP) and the dimensions as values.
            Embeddings may be represented as a matrix or a data frame
            (:py:class:`~numpy.ndarray`, :py:class:`~scipy.sparse.spmatrix`,
            :py:class:`~pandas.DataFrame` and :py:class:`~biocframe.BiocFrame.BiocFrame`).

        main_experiment_name (str, optional): Main experiment name.

        alternative_experiments (Dict[str, SummarizedExperiment], optional): Used to manage multi-modal experiments
            performed on the same sample/cells.

            Alternative experiments must contain the same cells (rows) as the primary experiment.
            It's a dictionary with keys as the names of the alternative experiments (e.g., sc-atac, crispr)
            and values as subclasses of :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

        row_pairs (MatrixTypesWithFrame, optional): Row pairings/relationships between features.

        col_pairs (MatrixTypesWithFrame, optional): Column pairings/relationships between cells.

        type_check_alternative_experiments (bool): Whether to strictly type check alternative experiments.
            All alternative experiments must be subclasses of
            :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.
    """

    def __init__(
        self,
        assays: Dict[str, MatrixTypes],
        row_ranges: Optional[GRangesOrGRangesList] = None,
        row_data: Optional[BiocOrPandasFrame] = None,
        col_data: Optional[BiocOrPandasFrame] = None,
        metadata: Optional[Dict] = None,
        reduced_dims: Optional[Dict[str, MatrixTypesWithFrame]] = None,
        main_experiment_name: Optional[str] = None,
        alternative_experiments: Optional[Dict[str, SummarizedExperiment]] = None,
        row_pairs: Optional[MatrixTypesWithFrame] = None,
        col_pairs: Optional[MatrixTypesWithFrame] = None,
        type_check_alternative_experiments: bool = True,
    ) -> None:
        """Initialize a single-cell experiment.

        Args:
            assays (Dict[str, MatrixTypes]): Dictionary of matrices with assay names as keys, and 2-dimensional matrices
                represented as :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

                Alternatively, you may use any 2-dimensional matrix that contains the property ``shape``
                and implements the slice operation using the ``__getitem__`` dunder method.

                All matrices in ``assays`` must be 2-dimensional and have the same shape
                (number of rows, number of columns).

            row_ranges (GRangesOrGRangesList, optional): Genomic features, must be the same length as rows of the 
                matrices in assays.

            row_data (BiocOrPandasFrame, optional): Features, must be the same length as rows of the matrices in assays.
                Features may be either a :py:class:`~pandas.DataFrame` or
                :py:class:`~biocframe.BiocFrame.BiocFrame`.

            col_data (BiocOrPandasFrame, optional): Sample data, must be the same length as columns of the matrices in
                assays. Sample Information may be either a :py:class:`~pandas.DataFrame` or
                :py:class:`~biocframe.BiocFrame.BiocFrame`.

            metadata (Dict, optional): Additional experimental metadata describing the methods.

            reduced_dims (Dict[str, MatrixTypesWithFrame], optional): Slot for lower dimensionality embeddings.

                Usually a dictionary with the embedding method as keys (e.g., t-SNE, UMAP) and the dimensions as values.
                Embeddings may be represented as a matrix or a data frame
                (:py:class:`~numpy.ndarray`, :py:class:`~scipy.sparse.spmatrix`,
                :py:class:`~pandas.DataFrame` and :py:class:`~biocframe.BiocFrame.BiocFrame`).

            main_experiment_name (str, optional): Main experiment name.

            alternative_experiments (Dict[str, SummarizedExperiment], optional): Used to manage multi-modal experiments
                performed on the same sample/cells.

                Alternative experiments must contain the same cells (rows) as the primary experiment.
                It's a dictionary with keys as the names of the alternative experiments (e.g., sc-atac, crispr)
                and values as subclasses of :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

            row_pairs (MatrixTypesWithFrame, optional): Row pairings/relationships between features.

            col_pairs (MatrixTypesWithFrame, optional): Column pairings/relationships between cells.

            type_check_alternative_experiments (bool): Whether to strictly type check alternative experiments.
                All alternative experiments must be subclasses of
                :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.
        """

        if row_ranges is None:
            row_ranges = GenomicRangesList(number_of_ranges=len(row_data))

        super().__init__(
            assays=assays,
            row_ranges=row_ranges,
            row_data=row_data,
            col_data=col_data,
            metadata=metadata,
        )

        self._set_rdims(reduced_dims=reduced_dims)

        self._main_experiment_name = main_experiment_name

        self._set_alt_expts(
            alt_expts=alternative_experiments,
            type_check_alternative_experiments=type_check_alternative_experiments,
        )

        self._set_row_pairs(row_pairs=row_pairs)
        self._set_col_pairs(col_pairs=col_pairs)

    def _validate_reduced_dims(self, reduced_dims: Dict[str, MatrixTypesWithFrame]):
        """Internal method to validate reduced dimensions. All dimensions must contain embeddings for all cells.

        Args:
            reduced_dims (Dict[str, MatrixTypesWithFrame]):
                Embeddings to validate.

        Raises:
            TypeError:
                If ``embeddings`` is not a matrix (numpy, scipy) or a pandas Dataframe.
                If ``reduced_dims`` is not a dictionary like object.
            ValueError: If length of dimensions do not match the number of cells.
        """
        if reduced_dims is not None:
            if not isinstance(reduced_dims, dict):
                raise TypeError("`reduced_dims` is not a dictionary.")

            for rdname, mat in reduced_dims.items():
                if not (hasattr(mat, "shape")):
                    raise TypeError(
                        f"Reduced dimension: '{rdname}' must be either a numpy ndarray, scipy "
                        "matrix, a pandas DataFrame or BiocFrame object."
                    )

                if self._shape[1] != mat.shape[0]:
                    raise ValueError(
                        f"Reduced dimension: '{rdname}' does not contain embeddings for all cells."
                    )

    def _validate_alternative_experiments(
        self,
        alternative_experiments: Optional[Dict[str, SummarizedExperiment]],
        type_check_alternative_experiments: bool,
    ):
        """Internal method to validate alternative experiments and optionally their types.

        Args:
            alternative_experiments (Dict[str, SummarizedExperiment], optional):
                A dictionary of alternative experiments with names as
                keys and values is a subclass of
                :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

                Note: The rows represent cells in the embeddings.

            type_check_alternative_experiments (bool): Whether to strictly type check alternative
                experiments.

        Raises:
            ValueError: If alternative experiments does not contain the same number of cells.
            TypeError: If alternative experiments is not a subclass of `SummarizedExperiment`.
        """
        if alternative_experiments is not None:
            if not isinstance(alternative_experiments, dict):
                raise TypeError("`alternative_experiments` is not a dictionary.")

            for alt_name, alternative_experiment in alternative_experiments.items():
                if self._shape[1] != alternative_experiment.shape[1]:
                    raise ValueError(
                        f"Alternative experiment: '{alt_name}' does not contain same number of"
                        " cells."
                    )

                if type_check_alternative_experiments is True:
                    if not issubclass(
                        type(alternative_experiment), SummarizedExperiment
                    ):
                        raise TypeError(
                            f"Alternative experiment: '{alt_name}' is not a subclass of"
                            " `SummarizedExperiment`."
                        )

    def _validate_pairs(self, pairs: Optional[MatrixTypesWithFrame]):
        """Validate row and column pairs.

        Currently only checks if they are dictionary like objects.

        Args:
            pairs (MatrixTypesWithFrame, optional): Pair to validate.

        Raises:
            TypeError: If pairs is not a dictionary like object.
        """
        if pairs is not None:
            if not isinstance(pairs, dict):
                raise TypeError("Pair is not a dictionary.")

    def _validate(self):
        """Internal method to validate the object."""
        super()._validate()

        self._validate_reduced_dims(self._reduced_dims)
        self._validate_alternative_experiments(
            self._alternative_experiments, self._type_check_alternative_experiments
        )

    def _set_rdims(self, reduced_dims):
        if reduced_dims is None:
            reduced_dims = {}

        self._validate_reduced_dims(reduced_dims)
        self._reduced_dims = reduced_dims

    def _set_alt_expts(self, alt_expts, type_check_alternative_experiments):
        if alternative_experiments is None:
            alternative_experiments = {}

        self._validate_alternative_experiments(
            alternative_experiments, type_check_alternative_experiments
        )
        self._type_check_alternative_experiments = type_check_alternative_experiments
        self._alternative_experiments = alternative_experiments

    def _set_row_pairs(self, row_pairs):
        if row_pairs is None:
            row_pairs = {}

        self._validate_pairs(row_pairs)
        self._row_pairs = row_pairs

    def _set_col_pairs(self, col_pairs):
        if col_pairs is None:
            col_pairs = {}

        self._validate_pairs(col_pairs)
        self._col_pairs = col_pairs

    @property
    def reduced_dims(
        self,
    ) -> Dict[str, MatrixTypesWithFrame]:
        """Access dimensionality embeddings.

        Returns:
            Dict[str, MatrixTypesWithFrame]: A dictionary of all embeddings, the embedding method as keys
            and values is an embedding.
        """
        return self._reduced_dims

    @reduced_dims.setter
    def reduced_dims(
        self,
        reduced_dims: Optional[Dict[str, MatrixTypesWithFrame]],
    ):
        """Set dimensionality embeddings.

        Args:
            reduced_dims (Dict[str, MatrixTypesWithFrame], optional): New embeddings to set.
                Pass None to remove the current embeddings.
        """
        self._set_rdims(reduced_dims)

    @property
    def main_experiment_name(self) -> Optional[str]:
        """Access main experiment name.

        Returns:
            (str, optional): Name if available, otherwise None.
        """
        return self._main_experiment_name

    @main_experiment_name.setter
    def main_experiment_name(self, name: Optional[str]):
        """Set main experiment name.

        Args:
            name (str, optional): Experiment name to set.
                Pass None to remove the current name.
        """
        self._main_experiment_name = name

    @property
    def reduced_dim_names(self) -> Optional[List[str]]:
        """Access names of dimensionality embeddings.

        Returns:
            (List[str], optional): List of all embeddings names if available.
        """

        if self.reduced_dims is not None:
            return list(self.reduced_dims.keys())

        return None

    def reduced_dim(self, name: str) -> MatrixTypesWithFrame:
        """Access an embedding by name.

        Args:
            name (str): Name of the embedding.

        Raises:
            ValueError: If embedding ``name`` does not exist.

        Returns:
            MatrixTypesWithFrame: The embedding represented as
            numpy, scipy matrix or a data frame from pandas or biocframe.
        """
        if name not in self._reduced_dims:
            raise ValueError(f"Embedding: '{name}' does not exist.")

        return self._reduced_dims[name]

    @property
    def alternative_experiments(
        self,
    ) -> Dict[str, SummarizedExperiment]:
        """Access alternative experiments.

        Returns:
            Dict[str, SummarizedExperiment]: A dictionary of alternative experiments with the name of
            the experiments as keys and value is a subclass of `SummarizedExperiment`.
        """
        return self._alternative_experiments

    @alternative_experiments.setter
    def alternative_experiments(
        self,
        alternative_experiments: Dict[str, SummarizedExperiment],
    ):
        """Set alternative experiments.

        Args:
            Dict[str, SummarizedExperiment]: New alternative experiments to set.
        """
        self._set_alt_expts(
            alt_expts=alternative_experiments,
            type_check_alternative_experiments=self._type_check_alternative_experiments,
        )

    def alternative_experiment(self, name: str) -> SummarizedExperiment:
        """Access alternative experiment by name.

        Args:
            name (str): Name of the alternative experiment.

        Raises:
            ValueError: If alternative experiment ``name`` does not exist.

        Returns:
            SummarizedExperiment: A `SummarizedExperiment`-like representation of the alternative
            experiment.
        """
        if name not in self._alternative_experiments:
            raise ValueError(f"Alternative experiment '{name}' does not exist.")

        return self._alternative_experiments[name]

    @property
    def row_pairs(self) -> Dict[str, MatrixTypesWithFrame]:
        """Access row pairings/relationships between features.

        Returns:
            Dict[str, MatrixTypesWithFrame]: Access row pairs.
        """
        return self._row_pairs

    @row_pairs.setter
    def row_pairs(self, pairs: Dict[str, MatrixTypesWithFrame]):
        """Set row pairs/relationships between features.

        Args:
            Dict[str, MatrixTypesWithFrame]: New row pairs to set.
        """
        self._set_row_pairs(pairs)

    @property
    def col_pairs(self) -> Dict[str, MatrixTypesWithFrame]:
        """Access column pairs/relationships between cells.

        Returns:
            Dict[str, MatrixTypesWithFrame]: Access column pairs.
        """
        return self._col_pairs

    @col_pairs.setter
    def col_pairs(self, pairs: Dict[str, MatrixTypesWithFrame]):
        """Set column pairs/relationships between cells.

        Args:
            Dict[str, MatrixTypesWithFrame]: New column pairs.
        """
        self._set_col_pairs(pairs)

    def __repr__(self) -> str:
        pattern = (
            f"Class SingleCellExperiment with {self.shape[0]} features and {self.shape[1]} cells \n"
            f"  main_experiment_name: {self.main_experiment_name if self.main_experiment_name is not None else None} \n"
            f"  assays: {list(self.assays.keys())} \n"
            f"  row_data: {self.row_data.columns if self.row_data is not None else None} \n"
            f"  col_data: {self.col_data.columns if self.col_data is not None else None} \n"
            f"  reduced_dims: {self.reduced_dim_names if self.reduced_dims is not None else None} \n"
            f"  alternative_experiments: {list(self.alternative_experiments.keys()) if self.alternative_experiments is not None else None}"  # noqa: E501
        )
        return pattern

    def __getitem__(self, args: SlicerArgTypes) -> "SingleCellExperiment":
        """Subset SingleCellExperiment.

        Note: Does not currently support slicing of
        :py:attr:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment.row_pairs`
        and
        :py:attr:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment.col_pairs`.

        Args:
            args (SlicerArgTypes): Indices or names to slice. The tuple contains
                slices along dimensions (rows, cols).

                Each element in the tuple, may be either a integer vector (index positions),
                boolean vector or :py:class:`~slice` object. Defaults to None.

        Raises:
            Exception: If too many or too few slices are provided.

        Returns:
            SingleCellExperiment: Sliced `SingleCellExperiment` object.
        """
        sliced_objs = self._slice(args)

        new_reduced_dims = None
        if self.reduced_dims is not None:
            new_reduced_dims = OrderedDict()
            for rdname, embeds in self.reduced_dims.items():
                sliced_embeds = None
                if isinstance(embeds, DataFrame):
                    sliced_embeds = embeds.iloc[sliced_objs.col_indices]
                else:
                    sliced_embeds = embeds[sliced_objs.col_indices, :]

                new_reduced_dims[rdname] = sliced_embeds

        new_alternative_experiments = None
        if self._alternative_experiments is not None:
            new_alternative_experiments = OrderedDict()
            for ae in self._alternative_experiments.keys():
                new_alternative_experiments[ae] = self._alternative_experiments[ae][
                    sliced_objs.row_indices, sliced_objs.col_indices
                ]

        return SingleCellExperiment(
            sliced_objs.assays,
            sliced_objs.row_data,
            sliced_objs.col_data,
            self.metadata,
            new_reduced_dims,
            self.main_experiment_name,
            new_alternative_experiments,
        )

    def to_anndata(self, alts: bool = False) -> Union[AnnData, Dict[str, AnnData]]:
        """Transform `SingleCellExperiment` object into :py:class:`~anndata.AnnData`.

        Args:
            alts (bool, optional): Whether to include alternative experiments
                int the result. Defaults to False.

        Returns:
            Union[AnnData, Dict[str, AnnData]]: A tuple with
            the main `AnnData` object and Optionally, If ``alts`` is true, a dictionary with
            alternative experiment names as keys and the value is the corresponding
            `AnnData` object.
        """

        from anndata import AnnData

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()

        trows = self.row_data
        if isinstance(self.row_data, GenomicRanges):
            trows = self.row_data.toPandas()

        obj = AnnData(
            obs=self.col_data,
            var=trows,
            uns=self.metadata,
            obsm=self.reduced_dims,
            layers=layers,
            varp=self.row_pairs,
            obsp=self.col_pairs,
        )

        if alts is True:
            adatas = None
            if self.alternative_experiments is not None:
                adatas = {}
                for (
                    alt_name,
                    alternative_experiment,
                ) in self.alternative_experiments.items():
                    adatas[alt_name] = alternative_experiment.to_anndata()

            return obj, adatas

        return obj

    def to_mudata(self) -> MuData:
        """Transform `SingleCellExperiment` object into :py:class:`~mudata.MuData`.

        If
        :py:attr:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment.main_experiment_name`
        is None, this experiment is called **Unknown Modality** in the :py:class:`~mudata.MuData`
        object.

        Returns:
            MuData: A MuData object.
        """

        from mudata import MuData

        mainData, altData = self.to_anndata(alts=True)

        expts = OrderedDict()
        mainName = self.main_experiment_name
        if self.main_experiment_name is None:
            mainName = "Unknown Modality"

        expts[mainName] = mainData

        if altData is not None:
            for exptName, expt in altData.items():
                expts[exptName] = expt

        return MuData(expts)
