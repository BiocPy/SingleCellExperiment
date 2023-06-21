import numpy as np
import pandas as pd
from singlecellexperiment import SingleCellExperiment
from singlecellexperiment import concat


__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


rowData1 = pd.DataFrame(
    {"feature_1": ["val_1", "val_2", "val_3"]}, index=["BRCA1", "PIK3CA", "ALK"]
)
rowData2 = pd.DataFrame(
    {
        "feature_1": ["val_1", "val_2", "val_3"],
        "feature_3": ["elem_1", "elem_2", "elem_3"],
    },
    index=["BRCA1", "HRAS", "ALK"],
)
rowData3 = pd.DataFrame(
    {
        "feature_1": ["val_1", "val_2", "val_3", "val_4"],
        "feature_2": ["rec_1", "rec_2", "rec_3", "rec_4"],
    },
    index=["BRCA1", "PIK3CA", "ALK", "RET proto-oncogene"],
)
colData1 = pd.DataFrame(
    {"meta_1": ["val_1", "val_2", "val_3"]}, index=["cell_1", "cell_2", "cell_3"]
)
colData2 = pd.DataFrame(
    {"meta_2": ["rec_1", "rec_2", "rec_3"], "meta_3": ["elem_1", "elem_2", "elem_3"]},
    index=["cell_4", "cell_5", "cell_6"],
)
colData3 = pd.DataFrame(
    {
        "meta_1": ["val_1", "val_2", "val_3", "val_4"],
        "meta_2": ["rec_1", "rec_2", "rec_3", "rec_4"],
        "meta_3": ["elem_1", "elem_2", "elem_3", "elem_4"],
    },
    index=["cell_7", "cell_8", "cell_9", "cell_10"],
)
assay1 = np.random.poisson(lam=5, size=(3, 3))
assay2 = np.random.poisson(lam=5, size=(3, 3))
assay3 = np.random.poisson(lam=5, size=(4, 4))

sce1 = SingleCellExperiment(
    assays={"counts": assay1}, rowData=rowData1, colData=colData1
)
sce2 = SingleCellExperiment(
    assays={"counts": assay2}, rowData=rowData2, colData=colData2
)
sce3 = SingleCellExperiment(
    assays={"counts": assay3}, rowData=rowData3, colData=colData3
)


def test_concat():
    sce_concat = concat(
        [sce1, sce2, sce3], mainExperimentName="test", metadata={"query": "an-awesome-query"}
    )
    assert sce_concat.shape == (5, 10)
    assert sce_concat.assays["counts"].shape == (5, 10)
    assert sce_concat.rowData.shape == (5, 3)
    assert sce_concat.colData.shape == (10, 3)
    assert sce_concat.mainExperimentName == "test"
    assert "query" in sce_concat.metadata
    assert sce_concat.metadata["query"] == "an-awesome-query"
