import anndata

from ..SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def fromAnnData(adata: anndata.AnnData) -> SingleCellExperiment:
    """Convert AnnData object to SingleCellExperiment

    Args:
        adata (AnnData): AnnData object

    Returns:
        SingleCellExperiment: single-cell experiment object
    """

    return SingleCellExperiment(
        assays=adata.layers,
        rows=adata.var,
        cols=adata.obs,
        metadata=adata.uns,
        reducedDims=adata.obsm,
        rowpairs=adata.varp,
        colpairs=adata.obsp,
    )


def fromH5AD(path: str) -> SingleCellExperiment:
    """Convert H5AD file to SingleCellExperiment

    Args:
        path (str): path to a H5AD file

    Returns:
        SingleCellExperiment: single-cell experiment object
    """

    adata = anndata.read_h5ad(path)
    return fromAnnData(adata)
