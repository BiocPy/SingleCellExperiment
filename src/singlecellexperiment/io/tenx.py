import h5py
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix

from ..SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def read_tenx_mtx(path: str) -> SingleCellExperiment:
    """Read 10X Matrix market directory as :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        path (str): Path to 10X MTX directory.
            Directory must contain `matrix.mtx`, and optionally
            a `genes.tsv` to represent featires and `barcodes.tsv` for cell
            annotations.

    Returns:
        SingleCellExperiment: A single-cell experiment object.
    """
    mat = mmread(f"{path}/matrix.mtx")
    mat = csr_matrix(mat)

    genes = pd.read_csv(path + "/genes.tsv", header=None, sep="\t")
    genes.columns = ["gene_ids", "gene_symbols"]

    cells = pd.read_csv(path + "/barcodes.tsv", header=None, sep="\t")
    cells.columns = ["barcode"]

    return SingleCellExperiment(assays={"counts": mat}, row_data=genes, col_data=cells)


def read_tenx_h5(path: str) -> SingleCellExperiment:
    """Read 10X H5 file as :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Note: Currently only supports version 3 of the 10X H5 format.

    Args:
        path (str): Path to 10x H5 file.

    Returns:
        SingleCellExperiment: A single-cell experiment object.
    """
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
    if "features" in groups:
        features = pd.DataFrame()
        for key, val in h5["matrix"]["features"].items():
            features[key] = [x.decode("ascii") for x in val[:]]

    barcodes = None
    if "barcodes" in groups:
        barcodes = pd.DataFrame()
        barcodes["barcodes"] = [x.decode("ascii") for x in h5["matrix"]["barcodes"][:]]

    return SingleCellExperiment(
        assays={"counts": counts}, row_data=features, col_data=barcodes
    )
