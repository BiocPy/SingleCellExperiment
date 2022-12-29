import h5py
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, csc_matrix

from ..SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def read10xMTX(path: str) -> SingleCellExperiment:
    """Read 10x Matrix market directory.

    Args:
        path (str): path to 10x mtx directory

    Returns:
        SingleCellExperiment: A `SingleCellExperiment` object
    """
    mat = mmread(f"{path}/matrix.mtx")
    mat = csr_matrix(mat)

    genes = pd.read_csv(path + "/genes.tsv", header=None, sep="\t")
    genes.columns = ["gene_ids", "gene_symbols"]

    cells = pd.read_csv(path + "/barcodes.tsv", header=None, sep="\t")
    cells.columns = ["barcode"]

    return SingleCellExperiment(assays={"counts": mat}, rowData=genes, colData=cells)


def read10xH5(path: str) -> SingleCellExperiment:
    """Read 10x H5 file. Must be V3 format.

    Args:
        path (str): path to 10x H5 file directory

    Returns:
        SingleCellExperiment: A `SingleCellExperiment` object
    """
    h5 = h5py.File(path, mode="r")

    if "matrix" not in h5.keys():
        raise ValueError(f"H5 file ({path}) is not a 10X v3 format.")

    groups = h5["matrix"].keys()

    # read the matrix
    data = h5["matrix"]["data"][:]
    indices = h5["matrix"]["indices"][:]
    indptr = h5["matrix"]["indptr"][:]
    shape = tuple(h5["matrix"]["shape"][:])

    print(shape)
    print(len(data), len(indices), len(indptr))
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
            features[key] = val[:]

    barcodes = None
    if "barcodes" in groups:
        barcodes = pd.DataFrame()
        barcodes["barcodes"] = h5["matrix"]["barcodes"][:]

    return SingleCellExperiment(
        assays={"counts": counts}, rowData=features, colData=barcodes
    )
