''' This version add the CTPS part
'''
from jumeg_causality import apply_inverse
import os, glob
method = 'dSPM'
subjects_dir = os.environ['SUBJECTS_DIR']
#evt_st = 'LRst'
##inverse epochs into the source space
#epo_st_list = glob.glob(subjects_dir+'/*/MEG/*evtW_%s_bc-epo.fif' %evt_st)
##epo_rt_list = glob.glob(subjects_dir+'/*/MEG/*evtW_%s_bc-epo.fif' %evt_rt)
#apply_inverse(epo_st_list[:], method=method, event=evt_st)
#from jumeg_causality import cal_labelts
evt_list = ['LLst','LRst', 'RRst', 'RLst']
#for event in evt_list:
#    stcs_path = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s/' %event)
#    cal_labelts(stcs_path, condition=event, min_subject='fsaverage')

#from jumeg_causality import normalize_data
#for event in evt_list:
#    fn_ts = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts.npy' %event)
#    normalize_data(fn_ts) 

#from jumeg_causality import model_estimation
#for event in evt_list:
#    fn_norm = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts,1-norm.npy' %event)
#    model_estimation(fn_norm)

#from jumeg_causality import causal_analysis
#for event in evt_list:
#    fn_norm = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts,1-norm.npy' %event)
#    causal_analysis(fn_norm)
#evt_list = ['LLst']
from jumeg_causality import sig_thresh, group_causality
#for event in evt_list:
#    fn_cau = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts,1-norm,cau.npy' %event)
#    sig_thresh(cau_list=fn_cau, condition=event, alpha=0.05)
    
for event in evt_list:
    fn_sig = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/sig_con/%s_sig_con_band.npy' %event)
    group_causality(fn_sig, event, submount=12)