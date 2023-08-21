# Tutorial

Container class to represent single-cell experiments; follows Bioconductor's [SingleCellExperiment](https://bioconductor.org/packages/release/bioc/html/SingleCellExperiment.html).

# Import as `SingleCellExperiment`

Readers are available to read `AnnData`, `H5AD` or `10x (MTX, H5) V3` formats as `SingleCellExperiment` objects.

```python
import singlecellexperiment

sce = singlecellexperiment.read_h5ad("tests/data/adata.h5ad")
```

Similarly `read_tenx_h5`, `read_tenx_mtx` and `from_anndata` methods are  available to read various formats.

# Construct a `SingleCellExperiment` object

`SingleCellExperiment` extends `SummarizedExperiment`, so it must contain `assays`, `row_data` and `column_data` objects. Additionally
`SingleCellExperiment` objects may contain dimensionality embeddings (e.g tSNE, UMAP etc), alternative experiment for multi-modal experiments and row/column pairings.

```python
import pandas as pd
import numpy as np
from genomicranges import GenomicRanges

nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
df_gr = pd.DataFrame(
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

gr = GenomicRanges.from_pandas(df_gr)

col_data = pd.DataFrame(
    {
        "celltype": ["cluster1", "cluster2"] * 3,
    }
)
```

Finally construct the object,

```python
from singlecellexperiment import SingleCellExperiment

tse = SingleCellExperiment(
    assays={"counts": counts}, row_data=df_gr, col_data=col_data
)
```

# Accessors

Multiple methods are available to access various slots of a `SingleCellExperiment` object

```python
tse.assays
tse.row_data
tse.col_data
tse.reduced_dims
tse.alternative_experiments
tse.row_pairs
tse.col_pairs
```

### Access specific sets

For reduced dimension and alternative experiment slots, one can also access specific objects

```python
tse.reduced_dim("tSNE")

tse.alternative_experiment("crop-seq")
```

# Subset experiment

Similar to `SummarizedExperiment`, you can subset by index

```python
# subset the first 10 rows and the first 3 samples
subset_tse = tse[0:10, 0:3]
```

# Export as AnnData objects

Methods are available to convert `SingleCellExperiment` objects as `AnnData`

```python
adata = tse.to_anndata()
```

or an `MuData`

```python
mdata = tse.to_mudata()
```
