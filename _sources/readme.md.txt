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

sce = singlecellexperiment.readH5AD("tests/data/adata.h5ad")
```

***OR construct one from scratch***

```python
from singlecellexperiment import SingleCellExperiment

tse = SingleCellExperiment(
    assays={"counts": counts}, rowData=df_gr, colData=colData,
    reducedDims={"tsne": ..., "umap": ...}, altExps={"atac": ...}
)
```

`SingleCellExperiment` extends `SummarizedExperiment`, so most methods from there are applicable here. checkout the [documentation](https://biocpy.github.io/SingleCellExperiment/).

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
