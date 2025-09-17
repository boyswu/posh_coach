
# Author: wujiahang
import numpy as np
from fastdtw import fastdtw
def align_dtw(ft, fu):
    if len(ft)==0 or len(fu)==0: return 0.0, []
    d, path = fastdtw(ft, fu, dist=lambda a,b: float(np.linalg.norm(a-b, ord=1)))
    return d, path