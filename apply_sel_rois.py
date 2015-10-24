import os, mne
import numpy as np
from dirs_manage import reset_directory
from jumeg.jumeg_preprocessing import get_files_from_list

def _cluster_rois(sel_path, label_list, stc, src):
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
            if test_label.hemi != class_label.hemi:
                i = i + 1
                continue
            else:
                class_pca = stc.extract_label_time_course(class_label, src, mode='pca_flip')
                test_pca = stc.extract_label_time_course(test_label, src, mode='pca_flip')
                class_pow = np.sum(class_pca ** 2)
                test_pow = np.sum(test_pca ** 2)
                max_pca = class_pca
                exch = False
                if class_pow < test_pow:
                    max_pca = test_pca
                    exch = True
                thre = max_pca.std() * 0.8
                diff =  np.abs(np.linalg.norm(class_pca) - np.linalg.norm(test_pca))
                if diff < thre:
                    if exch == True:
                        os.remove(class_list[i])
                        class_list[i] = test_fn
                    elif exch == False:
                        os.remove(test_fn)
                    belong = True
                i = i + 1
        if belong is False:
            class_list.append(test_fn)
                
    return len(class_list)


def sele_rois(labels_path_list, event, fn_stc, fn_src):
    """
    select ROIs, so that the overlapped lables merged into one. 
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
        sel_path = labels_path + '/ROIs/'        
        reset_directory(sel_path)
        source_path = labels_path + '/%s' %event
        for filename in glob.glob(os.path.join(source_path, '*.*')):
            shutil.copy(filename, sel_path)
        reducer = True
        stc = mne.read_source_estimate(fn_stc)
        src = mne.read_source_spaces(fn_src)
        while reducer:
            list_dirs = os.walk(sel_path)
            label_list = ['']
            for root, dirs, files in list_dirs:
                for f in files:
                    label_fname = os.path.join(root, f)
                    label_list.append(label_fname)
            label_list = label_list[1:]
            len_class = _cluster_rois(sel_path, label_list, stc, src)
            if len_class == len(label_list):
                reducer = False  
                
subjects_dir = os.environ['SUBJECTS_DIR']
method = 'dSPM'
event = 'LLst'
fn_stc = subjects_dir+'/fsaverage/%s_ROIs/%s-lh.stc' %(method,event)
fn_src = subjects_dir+'/fsaverage/bem/fsaverage-ico-5-src.fif'
labels_path = subjects_dir+'/fsaverage/%s_ROIs' %(method)                
sele_rois(labels_path_list=labels_path, event=event, fn_stc=fn_stc, fn_src=fn_src)