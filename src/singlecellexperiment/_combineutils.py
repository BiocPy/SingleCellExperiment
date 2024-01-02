import itertools

import biocutils as ut
import numpy as np

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def merge_generic(se, by, attr):
    if by not in ["row", "column"]:
        raise ValueError("'by' must be either 'row' or 'column'.")

    assays = [getattr(x, attr) for x in se]
    _all_keys = [list(x.keys()) for x in assays]
    _all_keys = list(set(itertools.chain.from_iterable(_all_keys)))

    _all_assays = {}
    for k in _all_keys:
        _all_mats = [x[k] for x in assays]

        if by == "row":
            _all_assays[k] = ut.combine_rows(*_all_mats)
        else:
            _all_assays[k] = ut.combine_columns(*_all_mats)

    return _all_assays


def relaxed_merge_generic(se, by, attr):
    if by not in ["row", "column"]:
        raise ValueError("'by' must be either 'row' or 'column'.")

    assays = [getattr(x, attr) for x in se]
    _all_keys = [list(x.keys()) for x in assays]
    _all_keys = list(set(itertools.chain.from_iterable(_all_keys)))

    _all_assays = {}
    for k in _all_keys:
        _all_mats = [x[k] for x in assays]

        if by == "row":
            _all_assays[k] = ut.relaxed_combine_rows(*_all_mats)
        else:
            _all_assays[k] = ut.relaxed_combine_columns(*_all_mats)

    return _all_assays


def relaxed_merge_numpy_generic(se, by, attr, names_attr):
    if by not in ["row", "column"]:
        raise ValueError("'by' must be either 'row' or 'column'.")

    _all_keys = [getattr(x, names_attr) for x in se]
    _all_keys = list(set(itertools.chain.from_iterable(_all_keys)))

    _all_assays = {}
    for k in _all_keys:
        _all_mats = []
        for x in se:
            _txmat = None
            if k not in getattr(x, names_attr):
                _txmat = np.ma.array(
                    np.zeros(shape=x.shape),
                    mask=True,
                )
            else:
                _txmat = getattr(x, attr)[k]

            _all_mats.append(_txmat)

        if by == "row":
            _all_assays[k] = ut.combine_rows(*_all_mats)
        else:
            _all_assays[k] = ut.combine_columns(*_all_mats)

    return _all_assays
