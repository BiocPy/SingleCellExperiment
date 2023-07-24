from typing import Union

import pandas as pd
from summarizedexperiment.types import MatrixTypes

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypesWithFrame = Union[MatrixTypes, pd.DataFrame]
