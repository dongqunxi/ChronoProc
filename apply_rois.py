import os, mne
import numpy as np
from dirs_manage import reset_directory
evt_st, evt_rt = 'LLst', 'LLrt'

subjects_dir = os.environ['SUBJECTS_DIR']
method = 'dSPM'
fn_avg = subjects_dir+'/fsaverage/%s_ROIs/%s-lh.stc' %(method,evt_st)
stc_avg = mne.read_source_estimate(fn_avg)
stc_avg = stc_avg.crop(0.1, 0.5)
src_pow = np.sum(stc_avg.data ** 2, axis=1)
stc_avg.data[src_pow < np.percentile(src_pow, 95)] = 0.
fn_src = subjects_dir+'/fsaverage/bem/fsaverage-ico-5-src.fif'
src_inv = mne.read_source_spaces(fn_src)
func_labels_lh, func_labels_rh = mne.stc_to_label(
                stc_avg, src=src_inv, smooth=True,
                subjects_dir=subjects_dir,
                connected=True)
# Left hemisphere definition
i = 0
labels_path = subjects_dir+'/fsaverage/%s_ROIs/%s/' %(method,evt_st)
reset_directory(labels_path)
while i < len(func_labels_lh):
    func_label = func_labels_lh[i]
    func_label.save(labels_path + '%s_%d' %(evt_st, i))
    i = i + 1
# right hemisphere definition
j = 0
while j < len(func_labels_rh):
    func_label = func_labels_rh[j]
    func_label.save(labels_path + '%s_%d' %(evt_st, j))
    j = j + 1

