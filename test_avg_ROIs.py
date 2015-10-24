from avg_ROIs_definition import apply_inverse, apply_stcs, apply_rois, apply_stand, apply_merge
from avg_ROIs_definition02 import sele_rois
import os, glob
import numpy as np

subjects_dir = os.environ['SUBJECTS_DIR']
fn_src = subjects_dir + '/fsaverage/bem/fsaverage-ico-5-src.fif'

evt_list = ['LLst', 'LRst', 'RLst', 'RRst']
tmin, tmax = 0.05, 0.45
for evt_st in evt_list:
    epo_list = glob.glob(subjects_dir+'/*/MEG/*evtW_%s_bc-epo.fif' %evt_st)
    fn_stc = subjects_dir + '/fsaverage/dSPM_ROIs/%s-lh.stc' %evt_st
    #apply_inverse(epo_list, event=evt_st, snr=5.0)
    #apply_stcs(method='dSPM', event=evt_st)
    #apply_rois(fn_stc, tmin=tmin, tmax=tmax, thr=95)
    sele_rois(fn_stc_list=fn_stc, fn_src=fn_src, min_dist=14, weight=0.9, tmin=tmin, tmax=tmax)
    apply_stand(fn_stc, radius=7.0, tmin=tmin, tmax=tmax)
    
evt_list = ['LLrt', 'LRrt', 'RLrt', 'RRrt']
tmin, tmax = -0.35, 0.05
for evt_st in evt_list:
    epo_list = glob.glob(subjects_dir+'/*/MEG/*evtW_%s_bc-epo.fif' %evt_st)
    fn_stc = subjects_dir + '/fsaverage/dSPM_ROIs/%s-lh.stc' %evt_st
    #apply_inverse(epo_list, event=evt_st, snr=5.0)
    #apply_stcs(method='dSPM', event=evt_st)
    #apply_rois(fn_stc, tmin=tmin, tmax=tmax, thr=95)
    sele_rois(fn_stc_list=fn_stc, fn_src=fn_src, min_dist=14, weight=0.9, tmin=tmin, tmax=tmax)
    apply_stand(fn_stc, radius=7.0, tmin=tmin, tmax=tmax)
evt_list = np.array([['LLst', 'LLrt'],['LRst', 'LRrt'],['RLst', 'RLrt'],['RRst', 'RRrt']])
labels_path = '/home/uais/data/Chrono/18subjects/fsaverage/dSPM_ROIs/'
apply_merge(labels_path, evt_list)