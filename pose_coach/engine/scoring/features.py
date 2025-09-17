# Author: wujiahang
import numpy as np
from .segment import Segmenter
from ..utils.geom import angle, rms, jerk_energy
def compute_angles(points):
    idx={'L_HIP':23,'R_HIP':24,'L_KNEE':25,'R_KNEE':26,'L_ANK':27,'R_ANK':28,'L_SH':11,'R_SH':12,'L_EL':13,'R_EL':14,'L_WR':15,'R_WR':16}
    A={}
    A['knee_L']=angle(points[idx['L_HIP'],:3], points[idx['L_KNEE'],:3], points[idx['L_ANK'],:3])
    A['knee_R']=angle(points[idx['R_HIP'],:3], points[idx['R_KNEE'],:3], points[idx['R_ANK'],:3])
    A['hip_L']= angle(points[idx['L_SH'],:3], points[idx['L_HIP'],:3], points[idx['L_KNEE'],:3])
    A['hip_R']= angle(points[idx['R_SH'],:3], points[idx['R_HIP'],:3], points[idx['R_KNEE'],:3])
    A['elbow_L']=angle(points[idx['L_SH'],:3], points[idx['L_EL'],:3], points[idx['L_WR'],:3])
    A['elbow_R']=angle(points[idx['R_SH'],:3], points[idx['R_EL'],:3], points[idx['R_WR'],:3])
    return A
def rom(series):
    s=np.asarray(series,dtype=float); 
    return float(np.max(s)-np.min(s)) if s.size else 0.0
def symmetry(series_L, series_R):
    sL=np.asarray(series_L); sR=np.asarray(series_R)
    n=min(sL.size,sR.size)
    if n==0: return 0.0
    return 1.0 - float(np.mean(np.abs(sL[:n]-sR[:n]))/180.0)
def trunk_stability(trunk_angles):
    if len(trunk_angles)==0: return 0.0
    import numpy as np
    dev=trunk_angles - np.nanmean(trunk_angles)
    return 1.0 - min(1.0, rms(dev)/30.0)
def smoothness_score(angle_series):
    e=jerk_energy(angle_series); return 1.0/(1.0+e)
def visibility_score(vis_series):
    if not vis_series: return 0.0
    import numpy as np
    return float(np.clip(np.nanmean(vis_series),0.0,1.0))
