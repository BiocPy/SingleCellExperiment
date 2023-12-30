from ..SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def read_h5ad(path: str) -> SingleCellExperiment:
    """Create a ``SingleCellExperiment`` from a H5AD file.

    Args:
        path:
            Path to a H5AD file.

    Returns:
        A ``SingleCellExperiment`` object.
    """
    import anndata

    adata = anndata.read_h5ad(path)
    return SingleCellExperiment.from_anndata(adata)
