[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/SingleCellExperiment.svg)](https://pypi.org/project/SingleCellExperiment/)
![Unit tests](https://github.com/BiocPy/SingleCellExperiment/actions/workflows/run-tests.yml/badge.svg)

# SingleCellExperiment

This package provides container class to represent single-cell experimental data as 2-dimensional matrices. In these matrices, the rows typically denote features or genomic regions of interest, while columns represent cells. In addition, a `SingleCellExperiment` (SCE) object may contain low-dimensionality embeddings, alternative experiments performed on same sample or set of cells. Follows Bioconductor's [SingleCellExperiment](https://bioconductor.org/packages/release/bioc/html/SingleCellExperiment.html).


## Install

To get started, install the package from [PyPI](https://pypi.org/project/singlecellexperiment/)

```bash
pip install singlecellexperiment
```

## Usage

The `SingleCellExperiment` extends [RangeSummarizedExperiment](https://github.com/BiocPy/SummarizedExperiment) and contains additional attributes:

- `reduced_dims`: Slot for low-dimensionality embeddings for each cell.
- `alternative_experiments`: Manages multi-modal experiments performed on the same sample or set of cells.
- `row_pairs` or `column_pairs`: Stores relationships between features or cells.

Readers are available to parse h5ad or `AnnData` objects to SCE:

```python
import singlecellexperiment

sce = singlecellexperiment.read_h5ad("tests/data/adata.h5ad")
```

    ## output
    class: SingleCellExperiment
    dimensions: (20, 30)
    assays(3): ['array', 'sparse', 'X']
    row_data columns(5): ['var_cat', 'cat_ordered', 'int64', 'float64', 'uint8']
    row_names(0):
    column_data columns(5): ['obs_cat', 'cat_ordered', 'int64', 'float64', 'uint8']
    column_names(0):
    main_experiment_name:
    reduced_dims(0): []
    alternative_experiments(0): []
    row_pairs(0): []
    column_pairs(0): []
    metadata(2): O_recarray nested

***OR construct one from scratch***

```python
from singlecellexperiment import SingleCellExperiment

tse = SingleCellExperiment(
    assays={"counts": counts}, row_data=df_gr, col_data=col_data,
    reduced_dims={"tsne": ..., "umap": ...}, alternative_experiments={"atac": ...}
)
```

Since `SingleCellExperiment` extends `RangeSummarizedExperiment`, most methods especially slicing and accessors are applicable here.
Checkout the [documentation](https://biocpy.github.io/SingleCellExperiment/) for more info.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
