# Tutorial

`SingleCellExperiment` is a container class to represent data from single-cell experiments. Methods are available to convert between AnnData and SCE, slots for lower dimentionality embeddings, feature and cell pairings etc. For more detailed description checkout the [Bioc SingleCellExperiment R package](https://bioconductor.org/packages/release/bioc/html/SingleCellExperiment.html))

## Mock data 

we first create a mock dataset of 200 rows and 6 columns, also adding a cell annotations.

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

gr = GenomicRanges.fromPandas(df_gr)

colData = pd.DataFrame(
    {
        "celltype": ["cluster1", "cluster2"] * 3,
    }
)
```

### `SingleCellExperiment`

```python
from singlecellexperiment import SingleCellExperiment

tse = SingleCellExperiment(
    assays={"counts": counts}, rowData=df_gr, colData=colData
)
```

### Accessors

Multiple methods are available to access various slots of a `SingleCellExperiment` object

```python
tse.assays()
tse.rowData()
tse.colData()
tse.ReducedDims()
tse.altExps()
tse.rowPairs()
tse.colPairs()
```

### Access specific sets

For reduced dimension and alternative experiment slots, one can also access specific keys

```python
tse.reducedDim("tSNE")

tse.altExp("crop-seq")
```

## Subset experiment

Currently, the package provides methods to subset by indices

```python
# subset the first 10 rows and the first 3 samples
subset_tse = tse[0:10, 0:3]
```

## Export and import AnnData objects

Methods are available to also transform `AnnData` objects to `SCE`

To import

```python
from singlecellexperiment import fromAnnData

tse = fromAnnData(<AnnData object>)
```

To export

```python
adata = tse.toAnnData()
```