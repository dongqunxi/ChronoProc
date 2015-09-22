import glob
from dirs_manage import reset_directory
from DICS_ROIs_Definition01 import apply_inverse, apply_rois, merge_rois, stan_rois, group_rois
evt_st, evt_rt = 'LLst', 'LLrt'
eid_st, eid_rt = 1, 8
fmin, fmax = 8, 12
stmin, stmax = 0.0, 0.6
ntmin, ntmax = -0.2, 0.0 
epo_st_list = glob.glob('/home/qdong/18subjects/*/MEG/*evtW_%s_bc-epo.fif' %evt_st)
apply_inverse(epo_st_list, event=evt_st, ctmin=stmin, ctmax=stmax, nctmin=ntmin, nctmax=ntmax,
                  fmin=fmin, fmax=fmax, min_subject='fsaverage', STCs=False)
stc_st_list = glob.glob('/home/qdong/18subjects/fsaverage/DICS_ROIs/*/*evtW_%s_bc_4_8-lh.stc' %evt_st)
apply_rois(stc_st_list,event=evt_st, tmin=stmin, tmax=stmax, tstep=0.05, window=0.2, 
               fmin=fmin, fmax=fmax)
labels_path = glob.glob('/home/qdong/18subjects/fsaverage/DICS_ROIs/*') 
merge_rois(labels_path, evelist=['LLst'])
stan_path = '/home/qdong/18subjects/fsaverage/DICS_ROIs/standard/'
reset_directory(stan_path)
stan_rois(stc_st_list, stan_path, size=8.0)
labels_path = '/home/qdong/18subjects/fsaverage/DICS_ROIs'
merge_rois(labels_path, group=True)
mer_path = '/home/qdong/18subjects/fsaverage/DICS_ROIs/merged/'
com_path = '/home/qdong/18subjects/fsaverage/DICS_ROIs/common/'
group_rois(am_sub=9, com_path=com_path, mer_path=mer_path)