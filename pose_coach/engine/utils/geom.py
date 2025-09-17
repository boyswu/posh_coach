# Author: wujiahang
import numpy as np
def angle(a,b,c,eps=1e-9):
    ab=a-b; cb=c-b
    nab=np.linalg.norm(ab)+eps; ncb=np.linalg.norm(cb)+eps
    cosv=np.clip(np.dot(ab,cb)/(nab*ncb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))
def rms(x):
    x=np.asarray(x,dtype=float); 
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0
def jerk_energy(series):
    x=np.asarray(series,dtype=float)
    if x.size<3: return 0.0
    d2=np.diff(x,n=2); return float(np.mean(d2**2))
