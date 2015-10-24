from avg_ROIs_definition import DICS_inverse, apply_stcs, apply_rois, sele_rois, apply_stand, apply_merge
import os, glob
import numpy as np

subjects_dir = os.environ['SUBJECTS_DIR']
fn_src = subjects_dir + '/fsaverage/bem/fsaverage-ico-5-src.fif'
eid_st, eid_rt = 1, 8
evt_list = ['LLst', 'LRst', 'RLst', 'RRst']
tmin, tmax = 0.05, 0.45
for evt_st in evt_list:
    epo_list = glob.glob(subjects_dir+'/*/MEG/*evtW_%s_bc-epo.fif' %evt_st)
    fn_stc = subjects_dir + '/fsaverage/DICS_ROIs/%s-lh.stc' %evt_st
    DICS_inverse(epo_list,event_id=eid_st, event=evt_st, ctmin=tmin, ctmax=tmax)
    apply_stcs(method='DICS', event=evt_st)
    apply_rois(fn_stc, tmin=tmin, tmax=tmax, thr=95)
    sele_rois(fn_stc_list=fn_stc, fn_src=fn_src, min_dist=20, weight=0.9, tmin=tmin, tmax=tmax)
    apply_stand(fn_stc, radius=10.0, tmin=tmin, tmax=tmax)