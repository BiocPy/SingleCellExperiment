from warnings import warn

from biocframe import BiocFrame, from_pandas

from ..SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def read_tenx_mtx(path: str) -> SingleCellExperiment:
    """Read 10X Matrix market directory as :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        path:
            Path to 10X MTX directory.

            Directory must contain `matrix.mtx`, and optionally
            a `genes.tsv` to represent featires and `barcodes.tsv` for cell
            annotations.

    Returns:
        A single-cell experiment object.
    """

    import pandas as pd
    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    mat = mmread(f"{path}/matrix.mtx")
    mat = csr_matrix(mat)

    genes = pd.read_csv(path + "/genes.tsv", header=None, sep="\t")
    genes.columns = ["gene_ids", "gene_symbols"]

    cells = pd.read_csv(path + "/barcodes.tsv", header=None, sep="\t")
    cells.columns = ["barcode"]

    return SingleCellExperiment(
        assays={"counts": mat},
        row_data=from_pandas(genes),
        column_data=from_pandas(cells),
    )


def read_tenx_h5(path: str) -> SingleCellExperiment:
    """Read 10X H5 file as :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Note: Currently only supports version 3 of the 10X H5 format.

    Args:
        path:
            Path to 10x H5 file.

    Returns:
        A single-cell experiment object.
    """

    import h5py
    from scipy.sparse import csc_matrix, csr_matrix

    h5 = h5py.File(path, mode="r")

    if "matrix" not in h5.keys():
        raise ValueError(f"H5 file ({path}) is not a 10X V3 format.")

    groups = h5["matrix"].keys()

    # read the matrix
    data = h5["matrix"]["data"][:]
    indices = h5["matrix"]["indices"][:]
    indptr = h5["matrix"]["indptr"][:]
    shape = tuple(h5["matrix"]["shape"][:])

    counts = None
    if len(indptr) == shape[1] + 1:
        counts = csc_matrix((data, indices, indptr), shape=shape)
    else:
        counts = csr_matrix((data, indices, indptr), shape=shape)

    # read features
    features = None
    ignore_list = []
    if "features" in groups:
        features = {}
        for key, val in h5["matrix"]["features"].items():
            temp_features = [x.decode("ascii") for x in val]

            if len(temp_features) != counts.shape[0]:
                ignore_list.append(key)
            else:
                features[key] = temp_features

        features = BiocFrame(features, number_of_rows=counts.shape[0])

    if len(ignore_list) > 0:
        warn(
            f"These columns from h5 are ignored - {', '.join(ignore_list)} because of "
            "inconsistent length with the count matrix."
        )

    barcodes = None
    if "barcodes" in groups:
        barcodes = {}
        barcodes["barcodes"] = [x.decode("ascii") for x in h5["matrix"]["barcodes"]]
        barcodes = BiocFrame(barcodes, number_of_rows=counts.shape[1])

    return SingleCellExperiment(
        assays={"counts": counts}, row_data=features, column_data=barcodes
    )
