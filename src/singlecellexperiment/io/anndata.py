from collections import OrderedDict

from biocframe import from_pandas

from ..SingleCellExperiment import SingleCellExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _to_normal_dict(obj):
    norm_obj = obj
    if len(norm_obj.keys()) == 0:
        norm_obj = None
    else:
        norm_obj = OrderedDict()
        for okey, oval in norm_obj.items():
            norm_obj[okey] = oval

    return norm_obj


def from_anndata(adata: "AnnData") -> SingleCellExperiment:
    """Read an :py:class:`~anndata.AnnData` into
    :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        adata (AnnData): Input data.

    Returns:
        SingleCellExperiment: A single-cell experiment object.
    """

    layers = OrderedDict()
    for asy, mat in adata.layers.items():
        layers[asy] = mat.transpose()

    if adata.X is not None:
        layers["X"] = adata.X.transpose()

    obsm = _to_normal_dict(adata.obsm)
    varp = _to_normal_dict(adata.varp)
    obsp = _to_normal_dict(adata.obsp)

    return SingleCellExperiment(
        assays=layers,
        row_data=from_pandas(adata.var),
        col_data=from_pandas(adata.obs),
        metadata=adata.uns,
        reduced_dims=obsm,
        row_pairs=varp,
        col_pairs=obsp,
    )


def read_h5ad(path: str) -> SingleCellExperiment:
    """Read a H5ad file as :py:class:`~singlecellexperiment.SingleCellExperiment.SingleCellExperiment`.

    Args:
        path (str): Path to a H5AD file.

    Returns:
        SingleCellExperiment: A single-cell experiment object.
    """

    import anndata

    adata = anndata.read_h5ad(path)
    return from_anndata(adata)
