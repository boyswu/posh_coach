# Author: wujiahang
import os, json, numpy as np
from .features import rom, symmetry, trunk_stability, smoothness_score, visibility_score, compute_angles
from .segment import Segmenter
def _map_linear(value, lo, hi, inv=False, clip=True):
    if hi==lo: return 0.0
    t=(value-lo)/(hi-lo)
    t=1.0-t if inv else t
    if clip: t=max(0.0, min(1.0, t))
    return float(t)
class ScoreAggregator:
    def __init__(self, action_name, config_dir):
        self.action_name=action_name
        cfg_path=os.path.join(config_dir, f"{action_name}.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config for action: {action_name}")
        self.cfg=json.load(open(cfg_path,'r',encoding='utf-8'))
        self.segmenter=Segmenter(self.cfg.get('segment_main_series','knee_L'),
                                 self.cfg.get('segment_min_prominence',10),
                                 self.cfg.get('segment_min_distance',8))
        self.angles_hist={k:[] for k in ['knee_L','knee_R','hip_L','hip_R','elbow_L','elbow_R','trunk']}
        self.visibility_hist=[]
    def add_frame(self, points33_xyz_visibility):
        ang=compute_angles(points33_xyz_visibility)
        trunk=0.5*(ang['hip_L']+ang['hip_R'])
        self.angles_hist['knee_L'].append(ang['knee_L']); self.angles_hist['knee_R'].append(ang['knee_R'])
        self.angles_hist['hip_L'].append(ang['hip_L']);   self.angles_hist['hip_R'].append(ang['hip_R'])
        self.angles_hist['elbow_L'].append(ang['elbow_L']); self.angles_hist['elbow_R'].append(ang['elbow_R'])
        self.angles_hist['trunk'].append(trunk)
        vis=float(np.nanmean(points33_xyz_visibility[:,3])); self.visibility_hist.append(vis)
        self.segmenter.add(self.angles_hist[self.cfg.get('segment_main_series','knee_L')][-1])
    def _score_rep(self, s,e):
        import numpy as np
        sl=slice(s,e)
        kneeL=np.array(self.angles_hist['knee_L'][sl]); kneeR=np.array(self.angles_hist['knee_R'][sl])
        hipL=np.array(self.angles_hist['hip_L'][sl]);   hipR=np.array(self.angles_hist['hip_R'][sl])
        elbowL=np.array(self.angles_hist['elbow_L'][sl]); elbowR=np.array(self.angles_hist['elbow_R'][sl])
        trunk=np.array(self.angles_hist['trunk'][sl]);  vis=np.array(self.visibility_hist[sl])
        C=self.cfg
        rom_knee=max(rom(kneeL), rom(kneeR)); s_rom=_map_linear(rom_knee, C['rom']['min'], C['rom']['max'])
        s_sym=symmetry(kneeL, kneeR); s_trunk=trunk_stability(trunk)
        s_smooth=smoothness_score(kneeL); L=len(kneeL)
        s_tempo=_map_linear(L, C['tempo']['min_len'], C['tempo']['max_len'], clip=True)
        base = (s_rom+s_sym+s_trunk+s_smooth+s_tempo)/5.0*100.0
        if rom_knee < C['hard']['rom_min'] or s_trunk < C['hard']['trunk_min']:
            base = min(base, C['hard']['cap_score'])
        W=C['weights']
        final=(s_rom*W['rom']+s_sym*W['symmetry']+s_trunk*W['trunk']+s_smooth*W['smoothness']+s_tempo*W['tempo'])/sum(W.values())
        s_vis=float(np.clip(np.mean(vis),0.0,1.0)) if vis.size>0 else 0.0
        final*=max(0.6, s_vis)
        return float(final*100.0), {'rom_knee':rom_knee,'symmetry':s_sym,'trunk':s_trunk,'smoothness':s_smooth,'tempo_len':L,'visibility':s_vis}
    def flush_completed_reps(self):
        out=[]
        for (s,e) in self.segmenter.emit_rep_indices():
            sc, detail = self._score_rep(s,e)
            out.append({'start':s,'end':e,'score':sc,'detail':detail})
        return out
