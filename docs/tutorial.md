---
file_format: mystnb
kernelspec:
  name: python
---

## Represent single-cell experiments

This package provides container class to represent single-cell experimental data as 2-dimensional matrices. In these matrices, the rows typically denote features or genomic regions of interest, while columns represent cells. In addition, a `SingleCellExperiment` (SCE) object may contain low-dimensionality embeddings, alternative experiments performed on same sample or set of cells.

:::{important}
The design of `SingleCellExperiment` class and its derivates adheres to the R/Bioconductor specification, where rows correspond to features, and columns represent cells.
:::

:::{note}
These classes follow a functional paradigm for accessing or setting properties, with further details discussed in [functional paradigm](https://biocpy.github.io/tutorial/chapters/philosophy.html#functional-discipline) section.
:::

## Installation

To get started, install the package from [PyPI](https://pypi.org/project/singlecellexperiment/)

```bash
pip install singlecellexperiment
```

## Construction

The `SingleCellExperiment` extends `RangeSummarizedExperiment` and contains additional attributes:

- `reduced_dims`: Slot for low-dimensionality embeddings for each cell.
- `alternative_experiments`: Manages multi-modal experiments performed on the same sample or set of cells.
- `row_pairs` or `column_pairs`: Stores relationships between features or cells.

:::{note}
In contrast to R, matrices in Python are unnamed and do not contain row or column names. Hence, these matrices cannot be directly used as values in assays or alternative experiments. We strictly enforce type checks in these cases. To relax these restrictions for alternative experiments, set `type_check_alternative_experiments` to `False`.
:::

:::{important}
If you are using the `alternative_experiments` slot, the number of cells must match the parent experiment.  Otherwise, the expectation is that the cells do not share the same sample or annotations and cannot be set in alternative experiments!
:::

Before we construct a `SingleCellExperiment` object, lets generate information about rows, columns and a mock experimental data from single-cell rna-seq experiments:

```{code-cell}

import pandas as pd
import numpy as np
from scipy import sparse as sp
from biocframe import BiocFrame
from genomicranges import GenomicRanges
from random import random

nrows = 200
ncols = 6
counts = sp.rand(nrows, ncols, density=0.2, format="csr")
row_data = BiocFrame(
    {
        "seqnames": [
            "chr1",
            "chr2",
            "chr2",
            "chr2",
            "chr1",
            "chr1",
            "chr3",
            "chr3",
            "chr3",
            "chr3",
        ]
        * 20,
        "starts": range(100, 300),
        "ends": range(110, 310),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 20,
        "score": range(0, 200),
        "GC": [random() for _ in range(10)] * 20,
    }
)

col_data = pd.DataFrame(
    {
        "celltype": ["cluster1", "cluster2"] * 3,
    }
)
```

Now lets create the `SingleCellExperiment` instance:

```{code-cell}
from singlecellexperiment import SingleCellExperiment

sce = SingleCellExperiment(
    assays={"counts": counts}, row_data=row_data, column_data=col_data,
    reduced_dims = {"random_embeds": np.random.rand(ncols, 4)}
)

print(sce)
```


:::{tip}
You can also use delayed or file-backed arrays for representing experimental data, check out [this section](./summarized_experiment.qmd#delayed-or-file-backed-arrays) from summarized experiment.
:::


### Interop with `anndata`

We provide convenient methods for loading an `AnnData` or `h5ad` file into `SingleCellExperiment` objects.

For example, lets create an `AnnData` object,

```{code-cell}
import anndata as ad
from scipy import sparse as sp

counts = sp.csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
adata = ad.AnnData(counts)
print(adata)
```

Converting `AnnData` as `SingleCellExperiment` is straightforward:

```{code-cell}
sce_adata = SingleCellExperiment.from_anndata(adata)
print(sce_adata)
```


and vice-verse.  All assays from SCE are represented in the `layers` slot of the `AnnData` object:

```{code-cell}
adata2 = sce_adata.to_anndata()
print(adata2)
```

Similarly, one can load a h5ad file:


```python
from singlecellexperiment import read_h5ad
sce_h5 = read_h5ad("../../assets/data/adata.h5ad")
print(sce_h5)
```

### From 10X formats

In addition, we also provide convenient methods to load a [10X Genomics HDF5 Feature-Barcode Matrix Format](https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-outputs-h5-matrices) file.

```python
from singlecellexperiment import read_tenx_h5
sce_h5 = read_tenx_h5("../../assets/data/tenx.sub.h5")
print(sce_h5)
```

:::{note}
Methods are also available to read a 10x matrix market directory using the `read_tenx_mtx` function.
:::

## Getters/Setters

Getters are available to access various attributes using either the property notation or functional style.

Since `SingleCellExperiment` extends `RangedSummarizedExperiment`, all getters and setters from the base class are accessible here; more details [here](./summarized_experiment.qmd).

```{code-cell}
# access assay names
print("reduced dim names (as property): ", sce.reduced_dim_names)
print("reduced dim names (functional style): ", sce.get_reduced_dim_names())

# access row data
print(sce.row_data)
```

#### Access a reduced dimension

One can access an reduced dimension by index or name:

```{code-cell}
sce.reduced_dim(0) # same as se.reduced_dim("random_embeds")
```

## Subset experiments

You can subset experimental data by using the subset (`[]`) operator. This operation accepts different slice input types, such as a boolean vector, a `slice` object, a list of indices, or names (if available) to subset.

In our previous example, we didn't include row or column names. Let's create another `SingleCellExperiment` object that includes names.

```{code-cell}
subset_sce = sce[0:10, 0:3]
print(subset_sce)
```


## Combining experiments

`SingleCellExperiment` implements methods for the `combine` generic from [**BiocUtils**](https://github.com/BiocPy/biocutils).

These methods enable the merging or combining of multiple `SingleCellExperiment` objects, allowing users to aggregate data from different experiments or conditions. Note: `row_pairs` and `column_pairs` are not ignored as part of this operation.


To demonstrate, let's create multiple `SingleCellExperiment` objects (read more about this in [combine section from `SummarizedExperiment`](./summarized_experiment.qmd#combining-experiments)).

```{code-cell}

ncols = 10
nrows = 100
sce1 = SingleCellExperiment(
    assays={"counts": np.random.poisson(lam=10, size=(nrows, ncols))},
    row_data=BiocFrame({"A": [1] * nrows}),
    column_data=BiocFrame({"A": [1] * ncols}),
)

sce2 = SingleCellExperiment(
    assays={
        "counts": np.random.poisson(lam=10, size=(nrows, ncols)),
        # "normalized": np.random.normal(size=(nrows, ncols)),
    },
    row_data=BiocFrame({"A": [3] * nrows}),
    column_data=BiocFrame({"A": [3] * ncols}),
)

rowdata1 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
coldata1 = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_2", "SAM_3"],
        "disease": ["True", "True", "True"],
        "doublet_score": [0.15, 0.62, 0.18],
    },
    index=["cell_1", "cell_2", "cell_3"],
)
sce_alts1 = SingleCellExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=rowdata1,
    column_data=coldata1,
    row_names=["HER2", "BRCA1", "TPFK"],
    column_names=["cell_1", "cell_2", "cell_3"],
    metadata={"seq_type": "paired"},
    reduced_dims={"PCA": np.random.poisson(lam=10, size=(3, 5))},
    alternative_experiments={
        "modality1": SingleCellExperiment(
            assays={"counts2": np.random.poisson(lam=10, size=(3, 3))},
        )
    },
)

rowdata2 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
coldata2 = pd.DataFrame(
    {
        "sample": ["SAM_4", "SAM_5", "SAM_6"],
        "disease": ["True", "False", "True"],
        "doublet_score": [0.05, 0.23, 0.54],
    },
    index=["cell_4", "cell_5", "cell_6"],
)
sce_alts2 = SingleCellExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        # "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=rowdata2,
    column_data=coldata2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
    reduced_dims={"PCA": np.random.poisson(lam=5, size=(3, 5))},
    alternative_experiments={
        "modality1": SingleCellExperiment(
            assays={"counts2": np.random.poisson(lam=5, size=(3, 3))},
        )
    },
)

```

The `combine_rows` or `combine_columns` operations, expect all experiments to contain the same assay names. To combine experiments by row:

```{code-cell}
from biocutils import relaxed_combine_columns, combine_columns, combine_rows, relaxed_combine_rows
sce_combined = combine_rows(sce2, sce1)
print(sce_combined)
```

Similarly to combine by column:

```{code-cell}
sce_combined = combine_columns(sce2, sce1)
print(sce_combined)
```

:::{note}
You can use `relaxed_combine_columns` or `relaxed_combined_rows` when there's mismatch in the number of features or samples. Missing rows or columns in any object are filled in with appropriate placeholder values before combining, e.g. missing assay's are replaced with a masked numpy array.
:::

```{code-cell}
# sce_alts1 contains an additional assay not present in sce_alts2
sce_relaxed_combine = relaxed_combine_columns(sce_alts1, sce_alts2)
print(sce_relaxed_combine)
```


## Export as `AnnData` or `MuData`

The package also provides methods to convert a `SingleCellExperiment` object into a `MuData` representation:

```{code-cell}
mdata = sce.to_mudata()
mdata
```

or coerce to an `AnnData` object:

```{code-cell}
adata, alts = sce.to_anndata()
print("main experiment: ", adata)
print("alternative experiments: ", alts)
```
