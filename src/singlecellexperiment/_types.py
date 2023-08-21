from typing import Union

from biocframe import BiocFrame
from pandas import DataFrame
from summarizedexperiment.types import MatrixTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypesWithFrame = Union[MatrixTypes, DataFrame, BiocFrame]
