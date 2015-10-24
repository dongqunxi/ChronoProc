import glob, os, mne
import numpy as np
from scipy.signal import detrend
from scipy.stats.mstats import zscore 
evt_st, evt_rt = 'LLst', 'LLrt'
tmin, tmax = 0.0, 0.6
subjects_dir = os.environ['SUBJECTS_DIR']
method = 'dSPM'
fn_list = glob.glob(subjects_dir+'/fsaverage/%s_ROIs/*/*,evtW_%s_bc-lh.stc' % (method, evt_st))
fn_avg = subjects_dir+'/fsaverage/%s_ROIs/%s' %(method,evt_st)
stcs = []
for fname in fn_list:
    stc = mne.read_source_estimate(fname)
    stc = stc.crop(tmin, tmax)
    cal_data = stc.data
    dt_data = detrend(cal_data, axis=-1)
    zc_data = zscore(dt_data, axis=-1)
    stc.data.setfield(zc_data, np.float32)
    stcs.append(stc)
stcs = np.array(stcs)
stc_avg = np.sum(stcs, axis=0)/stcs.shape[0]
stc_avg.save(fn_avg, ftype='stc')

        
        
        
        
        
