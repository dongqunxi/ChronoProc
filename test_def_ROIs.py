import mne, os
import numpy as np
from dirs_manage import reset_directory
subjects_dir = os.environ['SUBJECTS_DIR']
min_subject = 'fsaverage'
method = 'dSPM'
event = 'LLst'
tmin, tmax = 0.0, 0.6
window, tstep = 0.2, 0.05
thr = 99
fn_stc = '/home/uais/data/Chrono/18subjects/fsaverage/dSPM_ROIs/203731/203731_Chrono,nr,ocarta,evtW_LLst_bc-lh.stc'
stc_path = os.path.split(fn_stc)[0]
min_path = subjects_dir + '/%s' % min_subject
fn_src = min_path + '/bem/fsaverage-ico-5-src.fif'
# Make sure the target path is exist
labels_path = stc_path + '/%s/' %event
reset_directory(labels_path)
# Read the MNI source space
src_inv = mne.read_source_spaces(fn_src)
stc = mne.read_source_estimate(fn_stc, subject=min_subject)
stc = stc.crop(tmin, tmax)
src_pow = np.sum(stc.data ** 2, axis=1)
stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
tbeg = tmin
count = 1
while tbeg < tmax:
    tend = tbeg + window
    if tend > tmax:
        break
    win_stc = stc.copy().crop(tbeg, tend)
    stc_data = win_stc.data 
    src_pow = np.sum(stc_data ** 2, axis=1)
    win_stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
    func_labels_lh, func_labels_rh = mne.stc_to_label(
        win_stc, src=src_inv, smooth=True, connected=True,
        subjects_dir=subjects_dir)
    # Left hemisphere definition
    i = 0
    while i < len(func_labels_lh):
        func_label = func_labels_lh[i]
        func_label.save(labels_path + '%s_%s_win%d' % (event, str(i), count))
        i = i + 1
    # right hemisphere definition
    j = 0
    while j < len(func_labels_rh):
        func_label = func_labels_rh[j]
        func_label.save(labels_path +  '%s_%s_win%d' % (event, str(j), count))
        j = j + 1
    tbeg = tbeg + tstep
    count = count + 1