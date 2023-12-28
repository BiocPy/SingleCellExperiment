from collections import OrderedDict

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _to_normal_dict(obj):
    norm_obj = obj
    if len(norm_obj.keys()) == 0:
        norm_obj = None
    else:
        norm_obj = OrderedDict()
        for okey, oval in norm_obj.items():
            norm_obj[okey] = oval

    return norm_obj
