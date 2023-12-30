from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Union
from warnings import warn

import biocframe
import biocutils as ut
from genomicranges import GenomicRanges
from summarizedexperiment.RangedSummarizedExperiment import (
    GRangesOrGRangesList,
    RangedSummarizedExperiment,
)

from ._ioutils import _to_normal_dict

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _validate_reduced_dims(reduced_dims, shape):
    if reduced_dims is None:
        raise ValueError(
            "'reduced_dims' cannot be `None`, must be assigned to an empty dictionary."
        )

    if not isinstance(reduced_dims, dict):
        raise TypeError("'reduced_dims' is not a dictionary.")

    for rdname, mat in reduced_dims.items():
        if not hasattr(mat, "shape"):
            raise TypeError(
                f"Reduced dimension: '{rdname}' must be a matrix-like object."
                "Does not contain a `shape` property."
            )

        if shape[1] != mat.shape[0]:
            raise ValueError(
                f"Reduced dimension: '{rdname}' does not contain embeddings for all cells."
            )


def _validate_alternative_experiments(alternative_experiments, shape):
    if alternative_experiments is None:
        raise ValueError(
            "'alternative_experiments' cannot be `None`, must be assigned to an empty dictionary."
        )

    if not isinstance(alternative_experiments, dict):
        raise TypeError("'alternative_experiments' is not a dictionary.")

    for alt_name, alternative_experiment in alternative_experiments.items():
        if not hasattr(alternative_experiment, "shape"):
            raise TypeError(
                f"Alternative experiment: '{alt_name}' must be a 2-dimensional object."
                "Does not contain a `shape` property."
            )

        if shape[1] != alternative_experiment.shape[1]:
            raise ValueError(
                f"Alternative experiment: '{alt_name}' does not contain same number of"
                " cells."
            )


def _validate_pairs(pairs):
    if pairs is not None:
        if not isinstance(pairs, dict):
            raise TypeError("Pair is not a dictionary.")


class SingleCellExperiment(RangedSummarizedExperiment):
    """Container class for single-cell experiments, extending
    :py:class:`~summarizedexperiment.RangedSummarizedExperiment.RangedSummarizedExperiment` to provide slots for
    embeddings and alternative experiments that share the same cells.

    In contrast to R, :py:class:`~numpy.ndarray` or scipy matrices are unnamed and do
    not contain rownames and colnames. Hence, these matrices cannot be directly used as
    values in assays or alternative experiments. We strictly enforce type checks in these cases.

    To relax these restrictions for alternative experiments, set
    :py:attr:`~type_check_alternative_experiments` to `False`.

    If you are using the alternative experiment slot, the number of cells must match the
    parent experiment. Otherwise, these cells do not share the same sample or annotations
    and cannot be set in alternative experiments!

    Note: Validation checks do not apply to ``row_pairs`` or ``col_pairs``.
    """

    def __init__(
        self,
        assays: Dict[str, Any] = None,
        row_ranges: Optional[GRangesOrGRangesList] = None,
        row_data: Optional[biocframe.BiocFrame] = None,
        column_data: Optional[biocframe.BiocFrame] = None,
        row_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        reduced_dims: Optional[Dict[str, Any]] = None,
        main_experiment_name: Optional[str] = None,
        alternative_experiments: Optional[Dict[str, Any]] = None,
        row_pairs: Optional[Any] = None,
        column_pairs: Optional[Any] = None,
        validate: bool = True,
    ) -> None:
        """Initialize a single-cell experiment.

        Args:
            assays:
                A dictionary containing matrices, with assay names as keys
                and 2-dimensional matrices represented as either
                :py:class:`~numpy.ndarray` or :py:class:`~scipy.sparse.spmatrix`.

                Alternatively, you may use any 2-dimensional matrix that has
                the ``shape`` property and implements the slice operation
                using the ``__getitem__`` dunder method.

                All matrices in assays must be 2-dimensional and have the
                same shape (number of rows, number of columns).

            row_ranges:
                Genomic features, must be the same length as the number of rows of
                the matrices in assays.

            row_data:
                Features, must be the same length as the number of rows of
                the matrices in assays.

                Feature information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            column_data:
                Sample data, must be the same length as the number of
                columns of the matrices in assays.

                Sample information is coerced to a
                :py:class:`~biocframe.BiocFrame.BiocFrame`. Defaults to None.

            row_names:
                A list of strings, same as the number of rows.Defaults to None.

            column_names:
                A list of string, same as the number of columns. Defaults to None.

            metadata:
                Additional experimental metadata describing the methods.
                Defaults to None.

            reduced_dims:
                Slot for low-dimensionality embeddings.

                Usually a dictionary with the embedding method as keys (e.g., t-SNE, UMAP)
                and the dimensions as values.

                Embeddings may be represented as a matrix or a data frame, must contain a shape.

            main_experiment_name:
                A string, specifying the main experiment name.

            alternative_experiments:
                Used to manage multi-modal experiments performed on the same sample/cells.

                Alternative experiments must contain the same cells (rows) as the primary experiment.
                It's a dictionary with keys as the names of the alternative experiments
                (e.g., sc-atac, crispr) and values as subclasses of
                :py:class:`~summarizedexperiment.SummarizedExperiment.SummarizedExperiment`.

            row_pairs:
                Row pairings/relationships between features.

                Defaults to None.

            column_pairs:
                Column pairings/relationships between cells.

                Defaults to None.

            validate:
                Internal use only.
        """

        super().__init__(
            assays=assays,
            row_ranges=row_ranges,
            row_data=row_data,
            column_data=column_data,
            row_names=row_names,
            column_names=column_names,
            metadata=metadata,
            validate=validate,
        )
        self._main_experiment_name = main_experiment_name

        self._reduced_dims = reduced_dims if reduced_dims is not None else {}

        self._alternative_experiments = (
            alternative_experiments if alternative_experiments is not None else {}
        )

        self._row_pairs = row_pairs if row_pairs is not None else {}
        self._column_pairs = column_pairs if column_pairs is not None else {}

        if validate:
            _validate_reduced_dims(self._reduced_dims, self._shape)
            _validate_alternative_experiments(
                self._alternative_experiments, self._shape
            )
            _validate_pairs(self._row_pairs)
            _validate_pairs(self._column_pairs)

    #########################
    ######>> Copying <<######
    #########################

    def __deepcopy__(self, memo=None, _nil=[]):
        """
        Returns:
            A deep copy of the current ``SingleCellExperiment``.
        """
        from copy import deepcopy

        _assays_copy = deepcopy(self._assays)
        _rows_copy = deepcopy(self._rows)
        _rowranges_copy = deepcopy(self._row_ranges)
        _cols_copy = deepcopy(self._cols)
        _row_names_copy = deepcopy(self._row_names)
        _col_names_copy = deepcopy(self._column_names)
        _metadata_copy = deepcopy(self.metadata)
        _main_expt_name_copy = deepcopy(self._main_experiment_name)
        _red_dim_copy = deepcopy(self._reduced_dims)
        _alt_expt_copy = deepcopy(self._alternative_experiments)
        _row_pair_copy = deepcopy(self._row_pairs)
        _col_pair_copy = deepcopy(self._column_pairs)

        current_class_const = type(self)
        return current_class_const(
            assays=_assays_copy,
            row_ranges=_rowranges_copy,
            row_data=_rows_copy,
            column_data=_cols_copy,
            row_names=_row_names_copy,
            column_names=_col_names_copy,
            metadata=_metadata_copy,
            reduced_dims=_red_dim_copy,
            main_experiment_name=_main_expt_name_copy,
            alternative_experiments=_alt_expt_copy,
            row_pairs=_row_pair_copy,
            column_pairs=_col_pair_copy,
        )

    def __copy__(self):
        """
        Returns:
            A shallow copy of the current ``SingleCellExperiment``.
        """
        current_class_const = type(self)
        return current_class_const(
            assays=self._assays,
            row_ranges=self._row_ranges,
            row_data=self._rows,
            column_data=self._cols,
            row_names=self._row_names,
            column_names=self._column_names,
            metadata=self._metadata,
            reduced_dims=self._reduced_dims,
            main_experiment_name=self._main_experiment_name,
            alternative_experiments=self._alternative_experiments,
            row_pairs=self._row_pairs,
            column_pairs=self._column_pairs,
        )

    def copy(self):
        """Alias for :py:meth:`~__copy__`."""
        return self.__copy__()

    ##########################
    ######>> Printing <<######
    ##########################

    def __repr__(self) -> str:
        """
        Returns:
            A string representation.
        """
        output = f"{type(self).__name__}(number_of_rows={self.shape[0]}"
        output += f", number_of_columns={self.shape[1]}"
        output += ", assays=" + ut.print_truncated_list(self.assay_names)

        output += ", row_data=" + self._rows.__repr__()
        if self._row_names is not None:
            output += ", row_names=" + ut.print_truncated_list(self._row_names)

        output += ", column_data=" + self._cols.__repr__()
        if self._column_names is not None:
            output += ", column_names=" + ut.print_truncated_list(self._column_names)

        if self._row_ranges is not None:
            output += ", row_ranges=" + self._row_ranges.__repr__()

        if self._alternative_experiments is not None:
            output += ", alternative_experiments=" + ut.print_truncated_list(
                self.alternative_experiment_names
            )

        if self._reduced_dims is not None:
            output += ", reduced_dims=" + ut.print_truncated_list(
                self.reduced_dim_names
            )

        if self._main_experiment_name is not None:
            output += ", main_experiment_name=" + self._main_experiment_name

        if len(self._row_pairs) > 0:
            output += ", row_pairs=" + ut.print_truncated_dict(self._row_pairs)

        if len(self._column_pairs) > 0:
            output += ", column_pairs=" + ut.print_truncated_dict(self._column_pairs)

        if len(self._metadata) > 0:
            output += ", metadata=" + ut.print_truncated_dict(self._metadata)

        output += ")"
        return output

    def __str__(self) -> str:
        """
        Returns:
            A pretty-printed string containing the contents of this object.
        """
        output = f"class: {type(self).__name__}\n"

        output += f"dimensions: ({self.shape[0]}, {self.shape[1]})\n"

        output += f"assays({len(self.assay_names)}): {ut.print_truncated_list(self.assay_names)}\n"

        output += f"row_data columns({len(self._rows.column_names)}): {ut.print_truncated_list(self._rows.column_names)}\n"
        output += f"row_names({0 if self._row_names is None else len(self._row_names)}): {' ' if self._row_names is None else ut.print_truncated_list(self._row_names)}\n"

        output += f"column_data columns({len(self._cols.column_names)}): {ut.print_truncated_list(self._cols.column_names)}\n"
        output += f"column_names({0 if self._column_names is None else len(self._column_names)}): {' ' if self._column_names is None else ut.print_truncated_list(self._column_names)}\n"

        output += f"main_experiment_name: {' ' if self._main_experiment_name is None else self._main_experiment_name}\n"
        output += f"reduced_dims({len(self.reduced_dim_names)}): {ut.print_truncated_list(self.reduced_dim_names)}\n"
        output += f"alternative_experiments({len(self.alternative_experiment_names)}): {ut.print_truncated_list(self.alternative_experiment_names)}\n"
        output += f"row_pairs({len(self.row_pair_names)}): {ut.print_truncated_list(self.row_pair_names)}\n"
        output += f"column_pairs({len(self.column_pair_names)}): {ut.print_truncated_list(self.column_pair_names)}\n"

        output += f"metadata({str(len(self.metadata))}): {ut.print_truncated_list(list(self.metadata.keys()), sep=' ', include_brackets=False, transform=lambda y: y)}\n"

        return output

    ##############################
    ######>> reduced_dims <<######
    ##############################

    def get_reduced_dims(self) -> Dict[str, Any]:
        """Access dimensionality embeddings.

        Returns:
            A dictionary with keys as names of embedding method and value
            the embedding.
        """
        return self._reduced_dims

    def set_reduced_dims(
        self, reduced_dims: Dict[str, Any], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Set new reduced dimensions.

        Args:
            reduced_dims:
                New embeddings.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        _validate_reduced_dims(reduced_dims, self.shape)

        output = self._define_output(in_place)
        output._reduced_dims = reduced_dims
        return output

    @property
    def reduced_dims(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_reduced_dims`."""
        return self.get_reduced_dims()

    @reduced_dims.setter
    def reduced_dims(self, reduced_dims: Dict[str, Any]):
        """Alias for :py:meth:`~set_reduced_dims`."""
        warn(
            "Setting property 'reduced_dims' is an in-place operation, use 'set_reduced_dims' instead",
            UserWarning,
        )
        self.set_reduced_dims(reduced_dims, in_place=True)

    ####################################
    ######>> reduced_dims_names <<######
    ####################################

    def get_reduced_dim_names(self) -> List[str]:
        """Access reduced dimension names.

        Returns:
            List of reduced dimensionality names.
        """
        return list(self._reduced_dims.keys())

    def set_reduced_dim_names(
        self, names: List[str], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Replace :py:attr:`~.reduced_dims`'s names.

        Args:
            names:
                New dimensionality names.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        current_names = self.get_reduced_dim_names()
        if len(names) != len(current_names):
            raise ValueError(
                "Length of 'names' does not match the number of `reduced_dims`."
            )

        new_reduced_dims = OrderedDict()
        for idx in range(len(names)):
            new_reduced_dims[names[idx]] = self._reduced_dims.pop(current_names[idx])

        output = self._define_output(in_place)
        output._reduced_dims = new_reduced_dims
        return output

    @property
    def reduced_dim_names(self) -> List[str]:
        """Alias for :py:meth:`~get_reduced_dim_names`."""
        return self.get_reduced_dim_names()

    @reduced_dim_names.setter
    def reduced_dim_names(self, names: List[str]):
        """Alias for :py:meth:`~set_reduced_dim_names`."""
        warn(
            "Renaming names of property 'reduced_dims' is an in-place operation, use 'set_reduced_dim_names' instead",
            UserWarning,
        )
        self.set_reduced_dim_names(names, in_place=True)

    ####################################
    ######>> reduced_dim getter <<######
    ####################################

    def reduced_dim(self, dimension: Union[str, int]) -> Any:
        """Access an embedding by name.

        Args:
            dimension:
                Name or index position of the reduced dimension.

        Raises:
            AttributeError:
                If the dimension name does not exist.
            IndexError:
                If index is greater than the number of reduced dimensions.

        Returns:
            The embedding.
        """
        if isinstance(dimension, int):
            if dimension < 0:
                raise IndexError("Index cannot be negative.")

            if dimension > len(self.assay_names):
                raise IndexError("Index greater than the number of reduced dimensions.")

            return self._reduced_dims[self.reduced_dim_names[dimension]]
        elif isinstance(dimension, str):
            if dimension not in self._reduced_dims:
                raise AttributeError(f"Reduced dimension: {dimension} does not exist.")

            return self._reduced_dims[dimension]

        raise TypeError(
            f"'dimension' must be a string or integer, provided '{type(dimension)}'."
        )

    ################################
    ######>> main_expt_name <<######
    ################################

    def get_main_experiment_name(self) -> Optional[str]:
        """Access main experiment name.

        Returns:
            Name if available, otherwise None.
        """
        return self._main_experiment_name

    def set_main_experiment_name(
        self, name: Optional[str], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Set new experiment data (assays).

        Args:
            name:
                New main experiment name.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        output = self._define_output(in_place)
        output._main_experiment_name = name
        return output

    @property
    def main_experiment_name(self) -> Optional[str]:
        """Alias for :py:meth:`~get_main_experiment_name`."""
        return self.get_main_experiment_name()

    @main_experiment_name.setter
    def main_experiment_name(self, name: Optional[str]):
        """Alias for :py:meth:`~set_main_experiment_name`."""
        warn(
            "Setting property 'main_experiment_name' is an in-place operation, use 'set_main_experiment_name' instead",
            UserWarning,
        )
        self.set_main_experiment_name(name, in_place=True)

    #########################################
    ######>> alternative_experiments <<######
    #########################################

    def get_alternative_experiments(self) -> Dict[str, Any]:
        """Access alternative experiments.

        Returns:
            A dictionary with names of
            the experiments as keys and value the experiment.
        """
        return self._alternative_experiments

    def set_alternative_experiments(
        self, alternative_experiments: Dict[str, Any], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Set new alternative experiments.

        Args:
            alternative_experiments:
                New alternative experiments.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        _validate_alternative_experiments(alternative_experiments, self.shape)
        output = self._define_output(in_place)
        output._alternative_experiments = alternative_experiments
        return output

    @property
    def alternative_experiments(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_alternative_experiments`."""
        return self.get_alternative_experiments()

    @alternative_experiments.setter
    def alternative_experiments(self, alternative_experiments: Dict[str, Any]):
        """Alias for :py:meth:`~set_alternative_experiments`."""
        warn(
            "Setting property 'alternative_experiments' is an in-place operation, use 'set_alternative_experiments' instead",
            UserWarning,
        )
        self.set_alternative_experiments(alternative_experiments, in_place=True)

    ###############################################
    ######>> alternative_experiment_names <<######
    ###############################################

    def get_alternative_experiment_names(self) -> List[str]:
        """Access alternative experiment names.

        Returns:
            List of alternative experiment names.
        """
        return list(self._alternative_experiments.keys())

    def set_alternative_experiment_names(
        self, names: List[str], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Replace :py:attr:`~.alternative_experiment`'s names.

        Args:
            names:
                New alternative experiment names.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        current_names = self.get_alternative_experiment_names()
        if len(names) != len(current_names):
            raise ValueError(
                "Length of 'names' does not match the number of `alternative_experiments`."
            )

        new_alt_expts = OrderedDict()
        for idx in range(len(names)):
            new_alt_expts[names[idx]] = self._alternative_experiments.pop(
                current_names[idx]
            )

        output = self._define_output(in_place)
        output._alternative_experiments = new_alt_expts
        return output

    @property
    def alternative_experiment_names(self) -> List[str]:
        """Alias for :py:meth:`~get_alternative_experiment_names`."""
        return self.get_alternative_experiment_names()

    @alternative_experiment_names.setter
    def alternative_experiment_names(self, names: List[str]):
        """Alias for :py:meth:`~set_alternative_experiment_names`."""
        warn(
            "Renaming names of property 'alternative_experiments' is an in-place operation, use 'set_alternative_experiment_names' instead",
            UserWarning,
        )
        self.set_alternative_experiment_names(names, in_place=True)

    ###############################################
    ######>> alternative_experiment getter <<######
    ###############################################

    def alternative_experiment(self, name: Union[str, int]) -> Any:
        """Access alternative experiment by name.

        Args:
            name:
                Name or index position of the alternative experiment.

        Raises:
            AttributeError:
                If the dimension name does not exist.
            IndexError:
                If index is greater than the number of reduced dimensions.

        Returns:
            The alternative experiment.
        """
        if isinstance(name, int):
            if name < 0:
                raise IndexError("Index cannot be negative.")

            if name > len(self.assay_names):
                raise IndexError("Index greater than the number of reduced dimensions.")

            return self.alternative_experiments[self.alternative_experiment_names[name]]
        elif isinstance(name, str):
            if name not in self.reduced_dim:
                raise AttributeError(f"Reduced dimension: {name} does not exist.")

            return self.alternative_experiments[name]

        raise TypeError(f"'name' must be a string or integer, provided '{type(name)}'.")

    ###########################
    ######>> row_pairs <<######
    ###########################

    def get_row_pairs(self) -> Dict[str, Any]:
        """Access row pairings/relationships between features.

        Returns:
            Access row pairs.
        """
        return self._row_pairs

    def set_row_pairs(
        self, pairs: Dict[str, Any], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Replace :py:attr:`~.row_pairs`'s names.

        Args:
            names:
                New row pairs.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        _validate_pairs(pairs)

        output = self._define_output(in_place)
        output._row_pairs = pairs
        return output

    @property
    def row_pairs(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_row_pairs`."""
        return self.get_row_pairs()

    @row_pairs.setter
    def row_pairs(self, pairs: Dict[str, Any]):
        """Alias for :py:meth:`~set_row_pairs`."""
        warn(
            "Setting property 'row_pairs' is an in-place operation, use 'set_row_pairs' instead",
            UserWarning,
        )
        self.set_row_pairs(pairs, in_place=True)

    ####################################
    ######>> row_pairs_names <<######
    ####################################

    def get_row_pair_names(self) -> List[str]:
        """Access row pair names.

        Returns:
            List of row pair names.
        """
        return list(self._row_pairs.keys())

    def set_row_pair_names(
        self, names: List[str], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Replace :py:attr:`~.row_pair`'s names.

        Args:
            names:
                New names.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        current_names = self.get_row_pair_names()
        if len(names) != len(current_names):
            raise ValueError(
                "Length of 'names' does not match the number of `row_pairs`."
            )

        new_row_pairs = OrderedDict()
        for idx in range(len(names)):
            new_row_pairs[names[idx]] = self._row_pairs.pop(current_names[idx])

        output = self._define_output(in_place)
        output._row_pairs = new_row_pairs
        return output

    @property
    def row_pair_names(self) -> List[str]:
        """Alias for :py:meth:`~get_row_pair_names`."""
        return self.get_row_pair_names()

    @row_pair_names.setter
    def row_pair_names(self, names: List[str]):
        """Alias for :py:meth:`~set_row_pair_names`."""
        warn(
            "Renaming names of property 'row_pairs' is an in-place operation, use 'set_row_pair_names' instead",
            UserWarning,
        )
        self.set_row_pair_names(names, in_place=True)

    ##############################
    ######>> column_pairs <<######
    ##############################

    def get_column_pairs(self) -> Dict[str, Any]:
        """Access column pairings/relationships between cells.

        Returns:
            Access column pairs.
        """
        return self._column_pairs

    def set_column_pairs(
        self, pairs: Dict[str, Any], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Replace :py:attr:`~.column_pairs`'s names.

        Args:
            names:
                New column pairs.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        _validate_pairs(pairs)

        output = self._define_output(in_place)
        output._column_pairs = pairs
        return output

    @property
    def column_pairs(self) -> Dict[str, Any]:
        """Alias for :py:meth:`~get_column_pairs`."""
        return self.get_column_pairs()

    @column_pairs.setter
    def column_pairs(self, pairs: Dict[str, Any]):
        """Alias for :py:meth:`~set_column_pairs`."""
        warn(
            "Setting property 'column_pairs' is an in-place operation, use 'set_column_pairs' instead",
            UserWarning,
        )
        self.set_column_pairs(pairs, in_place=True)

    ####################################
    ######>> column_pairs_names <<######
    ####################################

    def get_column_pair_names(self) -> List[str]:
        """Access column pair names.

        Returns:
            List of column pair names.
        """
        return list(self._column_pairs.keys())

    def set_column_pair_names(
        self, names: List[str], in_place: bool = False
    ) -> "SingleCellExperiment":
        """Replace :py:attr:`~.column_pair`'s names.

        Args:
            names:
                New names.

            in_place:
                Whether to modify the ``SingleCellExperiment`` in place.

        Returns:
            A modified ``SingleCellExperiment`` object, either as a copy of the original
            or as a reference to the (in-place-modified) original.
        """
        current_names = self.get_column_pair_names()
        if len(names) != len(current_names):
            raise ValueError(
                "Length of 'names' does not match the number of `column_pairs`."
            )

        new_column_pairs = OrderedDict()
        for idx in range(len(names)):
            new_column_pairs[names[idx]] = self._column_pairs.pop(current_names[idx])

        output = self._define_output(in_place)
        output._column_pairs = new_column_pairs
        return output

    @property
    def column_pair_names(self) -> List[str]:
        """Alias for :py:meth:`~get_column_pair_names`."""
        return self.get_column_pair_names()

    @column_pair_names.setter
    def column_pair_names(self, names: List[str]):
        """Alias for :py:meth:`~set_column_pair_names`."""
        warn(
            "Renaming names of property 'column_pairs' is an in-place operation, use 'set_column_pair_names' instead",
            UserWarning,
        )
        self.set_column_pair_names(names, in_place=True)

    ##########################
    ######>> slicers <<#######
    ##########################

    # rest of them are inherited from BaseSE.

    def get_slice(
        self,
        rows: Optional[Union[str, int, bool, Sequence]],
        columns: Optional[Union[str, int, bool, Sequence]],
    ) -> "SingleCellExperiment":
        """Alias for :py:attr:`~__getitem__`, for back-compatibility."""

        slicer = self._generic_slice(rows=rows, columns=columns)

        new_row_ranges = None
        if slicer.row_indices != slice(None):
            new_row_ranges = self.row_ranges[slicer.row_indices]

        new_reduced_dims = {}
        for rdim, rmat in self._reduced_dims.items():
            if slicer.row_indices != slice(None):
                rmat = rmat[slicer.row_indices, :]

            if slicer.col_indices != slice(None):
                rmat = rmat[:, slicer.col_indices]

            new_reduced_dims[rdim] = rmat

        new_alt_expts = {}
        for altname, altexpt in self._alternative_experiments.items():
            if slicer.row_indices != slice(None):
                altexpt = altexpt[rows, :]

            if slicer.col_indices != slice(None):
                altexpt = altexpt[:, slicer.col_indices]

            new_alt_expts[altname] = altexpt

        new_row_pairs = {}
        for rname, rpair in self._row_pairs.items():
            if slicer.row_indices != slice(None):
                rpair = rpair[slicer.row_indices, :]

            if slicer.col_indices != slice(None):
                rpair = rpair[:, slicer.col_indices]

            new_row_pairs[rname] = rpair

        new_col_pairs = {}
        for cname, cpair in self._column_pairs.items():
            if slicer.row_indices != slice(None):
                cpair = cpair[slicer.row_indices, :]

            if slicer.col_indices != slice(None):
                cpair = cpair[:, slicer.col_indices]

            new_col_pairs[cname] = cpair

        current_class_const = type(self)
        return current_class_const(
            assays=slicer.assays,
            row_ranges=new_row_ranges,
            row_data=slicer.rows,
            column_data=slicer.columns,
            row_names=slicer.row_names,
            column_names=slicer.column_names,
            metadata=self._metadata,
            main_experiment_name=self._main_experiment_name,
            reduced_dims=new_reduced_dims,
            alternative_experiments=new_alt_expts,
            row_pairs=new_row_pairs,
            column_pairs=new_col_pairs,
        )

    ################################
    ######>> AnnData interop <<#####
    ################################

    def to_anndata(self, include_alternative_experiments: bool = False):
        """Transform ``SingleCellExperiment``-like into a :py:class:`~anndata.AnnData` representation.

        Args:
            include_alternative_experiments:
                Whether to transform alternative experiments.

        Returns:
            A tuple with ``AnnData`` main experiment and a list of alternative experiments.
        """
        from anndata import AnnData

        layers = OrderedDict()
        for asy, mat in self.assays.items():
            layers[asy] = mat.transpose()

        trows = self.row_data
        if isinstance(self.row_data, GenomicRanges):
            trows = self.row_data.to_pandas()

        obj = AnnData(
            obs=self.col_data,
            var=trows,
            uns=self.metadata,
            obsm=self.reduced_dims,
            layers=layers,
            varp=self.row_pairs,
            obsp=self.column_pairs,
        )

        if include_alternative_experiments is True:
            adatas = None
            if self.alternative_experiments is not None:
                adatas = {}
                for (
                    alt_name,
                    alternative_experiment,
                ) in self.alternative_experiments.items():
                    adatas[alt_name] = alternative_experiment.to_anndata()

            return obj, adatas

        return obj, None

    @classmethod
    def from_anndata(cls, input: "anndata.AnnData") -> "SingleCellExperiment":
        """Create a ``SingleCellExperiment`` from :py:class:`~anndata.AnnData`.

         Args:
            input:
                Input data.

        Returns:
            A ``SingleCellExperiment`` object.
        """

        layers = OrderedDict()
        for asy, mat in input.layers.items():
            layers[asy] = mat.transpose()

        if input.X is not None:
            layers["X"] = input.X.transpose()

        obsm = _to_normal_dict(input.obsm)
        varp = _to_normal_dict(input.varp)
        obsp = _to_normal_dict(input.obsp)

        return cls(
            assays=layers,
            row_data=biocframe.BiocFrame.from_pandas(input.var),
            column_data=biocframe.BiocFrame.from_pandas(input.obs),
            metadata=input.uns,
            reduced_dims=obsm,
            row_pairs=varp,
            column_pairs=obsp,
        )

    ###############################
    ######>> MuData interop <<#####
    ###############################

    def to_mudata(self):
        """Transform ``SingleCellExperiment`` object into :py:class:`~mudata.MuData` representation.

        If :py:attr:`~main_experiment_name` is `None`, this experiment is called
        **Unknown Modality** in the :py:class:`~mudata.MuData` object.

        Returns:
            A MuData object.
        """

        from mudata import MuData

        main_data, alt_data = self.to_anndata(include_alternative_experiments=True)

        expts = OrderedDict()
        mainName = self.main_experiment_name
        if self.main_experiment_name is None:
            mainName = "Unknown Modality"

        expts[str(mainName)] = main_data

        if alt_data is not None:
            for exptName, expt in alt_data.items():
                expts[str(exptName)] = expt

        return MuData(expts)
