# Author: wujiahang
import numpy as np
from scipy.signal import find_peaks
class Segmenter:
    def __init__(self, main_series_name='knee_L', min_prominence=10, min_distance=8):
        self.main_series_name=main_series_name
        self.buf=[]; self.min_prominence=min_prominence; self.min_distance=min_distance
    def add(self, value): self.buf.append(value)
    def emit_rep_indices(self):
        if len(self.buf)<20: return []
        arr=np.asarray(self.buf); inv=-arr
        peaks,_=find_peaks(inv, prominence=self.min_prominence, distance=self.min_distance)
        reps=[]
        for i in range(1,len(peaks)):
            s=int(peaks[i-1]); e=int(peaks[i])
            if e-s>=self.min_distance: reps.append((s,e))
        if len(peaks)>0:
            last=int(peaks[-1]); self.buf=self.buf[last:]
        return reps
