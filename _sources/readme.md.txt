# SingleCellExperiment

Container class to represent single-cell experiments; follows Bioconductor's [SingleCellExperiment](https://bioconductor.org/packages/release/bioc/html/SingleCellExperiment.html).


## Install

Package is published to [PyPI](https://pypi.org/project/singlecellexperiment/)

```shell
pip install singlecellexperiment
```

## Usage

Readers are available to read AnnData, H5AD or 10x (MTX, H5) V3 formats as `SingleCellExperiment` objects.

```python
import singlecellexperiment

sce = singlecellexperiment.read_h5ad("tests/data/adata.h5ad")
```

***OR construct one from scratch***

```python
from singlecellexperiment import SingleCellExperiment

tse = SingleCellExperiment(
    assays={"counts": counts}, row_data=df_gr, col_data=col_data,
    reduced_dims={"tsne": ..., "umap": ...}, alternative_experiments={"atac": ...}
)
```

Since `SingleCellExperiment` extends `SummarizedExperiment`, most methods especially slicing and accessors are applicable here. Checkout the [documentation](https://biocpy.github.io/SingleCellExperiment/) for more info.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
