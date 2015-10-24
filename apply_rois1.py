import os, mne
import numpy as np
from dirs_manage import reset_directory
from jumeg.jumeg_preprocessing import get_files_from_list

def apply_rois(fn_stc, event, tmin=0.0, tmax=0.3, tstep=0.05, window=0.2, 
                min_subject='fsaverage', thr=99):
    """
    Compute regions of interest (ROI) based on events
    ----------
    fn_stc : string
        evoked and morphed STC.
    event: string
        event of the related STC.
    tmin, tmax: float
        segment for ROIs definition.
    min_subject: string
        the subject as the common brain space.
    thr: float or int
        threshold of STC used for ROI identification.
    """
    from scipy.signal import detrend
    from scipy.stats.mstats import zscore 
    fnlist = get_files_from_list(fn_stc)
    # loop across all filenames
    for ifn_stc in fnlist:
        subjects_dir = os.environ['SUBJECTS_DIR']
        # extract the subject infromation from the file name
        stc_path = os.path.split(ifn_stc)[0]
        #name = os.path.basename(fn_stc)
        #tri = name.split('_')[1].split('-')[0]
        min_path = subjects_dir + '/%s' % min_subject
        fn_src = min_path + '/bem/fsaverage-ico-5-src.fif'
        # Make sure the target path is exist
        labels_path = stc_path + '/%s/' %event
        reset_directory(labels_path)
        # Read the MNI source space
        src_inv = mne.read_source_spaces(fn_src)
        stc = mne.read_source_estimate(ifn_stc, subject=min_subject)
        stc = stc.crop(tmin, tmax)
        src_pow = np.sum(stc.data ** 2, axis=1)
        stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
        #stc = stc.crop(tmin, tmax)
        #cal_data = stc.data
        #dt_data = detrend(cal_data, axis=-1)
        #zc_data = zscore(dt_data, axis=-1)
        #src_pow = np.sum(zc_data ** 2, axis=1)
        #stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
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
                win_stc, src=src_inv, smooth=True,
                subjects_dir=subjects_dir,
                connected=True)
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
            
def _cluster_rois(mer_path, label_list, count):
    """
    subfunctions of merge_ROIs
    ----------
    mer_path: str
        The directory for storing merged ROIs.
    label_list: list
        Labels to be merged
    """
    class_list = []
    class_list.append(label_list[0])
    for test_fn in label_list[1:]:
        test_label = mne.read_label(test_fn)
        i = 0
        belong = False
        while (i < len(class_list)) and (belong is False):
            class_label = mne.read_label(class_list[i])
            label_name = class_label.name
            if test_label.hemi != class_label.hemi:
                i = i + 1
                continue
            overlapped = len(np.intersect1d(test_label.vertices,
                                            class_label.vertices))
            if overlapped > 0:
                com_label = test_label + class_label
                pre_test = test_label.name.split('_')[0]
                pre_class = class_label.name.split('_')[0]
                #label_name = pre_class + '_%s-%s' %(pre_test,class_label.name.split('-')[-1])
                if pre_test != pre_class:
                    pre_class += ',%s' % pre_test
                    pre_class = list(set(pre_class.split(',')))
                    new_pre = ''
                    for pre in pre_class[:-1]:
                        new_pre += '%s,' % pre
                    new_pre += pre_class[-1]
                    label_name = '%s_%d_' % (new_pre, count) + \
                        class_label.name.split('_')[-1]
                os.remove(class_list[i])
                os.remove(test_fn)
                fn_newlabel = mer_path + '%s.label' %label_name
                if os.path.isfile(fn_newlabel):
                    fn_newlabel = fn_newlabel[:fn_newlabel.rfind('_')] + '_new,%s' %fn_newlabel.split('_')[-1]
                mne.write_label(fn_newlabel, com_label)
                class_list[i] = fn_newlabel
                belong = True
            i = i + 1
        if belong is False:
            class_list.append(test_fn)
    return len(class_list)


def merge_rois(labels_path_list, event='LLst'):
    """
    merge ROIs, so that the overlapped lables merged into one. 
    If 'group' is False, ROIs from all the events are merged and 
    saved in the folder 'ROIs' under the 'labels_path'.
    If 'group' is True, ROIs from all the subjects are merged and
    saved in the folder 'merged' under the 'labels_path'.
    ----------
    labels_path: the total path of all the ROIs' folders.
    group: if 'group' is False, merge ROIs from different events within one
           subject, if 'group' is True, merge ROIs across subjects.
    evelist: events name of all subfolders
    """
    path_list = get_files_from_list(labels_path_list)
    # loop across all filenames
    for labels_path in path_list:
        import glob, shutil
     
        mer_path = labels_path + '/ROIs/'
        reset_directory(mer_path)
        source_path = labels_path + '/%s' %event
        for filename in glob.glob(os.path.join(source_path, '*.*')):
            shutil.copy(filename, mer_path)

        # Merge the individual subject's ROIs
        reducer = True
        count = 1
        while reducer:
            list_dirs = os.walk(mer_path)
            label_list = ['']
            for root, dirs, files in list_dirs:
                for f in files:
                    label_fname = os.path.join(root, f)
                    label_list.append(label_fname)
            label_list = label_list[1:]
            len_class = _cluster_rois(mer_path, label_list, count)
            if len_class == len(label_list):
                reducer = False  
            count = count + 1
#subject_id = 'fsaverage'
#list_dirs = os.walk(labels_path) 
#lh_mvs = []
#rh_mvs = []
#for root, dirs, files in list_dirs: 
#    for f in files:
#        label_fname = os.path.join(root, f) 
#        label = mne.read_label(label_fname)
#        pca = stc_avg.extract_label_time_course(label, src_inv, mode='pca_flip')