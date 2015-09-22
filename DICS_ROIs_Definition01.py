from jumeg.jumeg_preprocessing import get_files_from_list
import mne
import os
import numpy as np
from dirs_manage import reset_directory, set_directory

 #######################################################
#                                                     #
# interface for creating the noise-covariance matrix  #
#                                                     #
#######################################################

   
def apply_inverse(fn_epo, event='LLst',ctmin=0.05, ctmax=0.25, nctmin=-0.2, nctmax=0,
                  fmin=4, fmax=8, min_subject='fsaverage', STCs=False):
    """
    Inverse evokes into source space using DICS method.
    ----------
    fn_epo : epochs of raw data.
    event_id: event id related with epochs.
    ctmin: the min time for computing CSD
    ctmax: the max time for computing CSD
    fmin: min value of the interest frequency band
    fmax: max value of the interest frequency band 
    min_subject: the subject for the common brain space.
    STCs: bool, make STCs of epochs.
    """
    from mne import Epochs, pick_types
    from mne.io import Raw
    from mne.event import make_fixed_length_events
    fnlist = get_files_from_list(fn_epo)
    # loop across all filenames
    for fname in fnlist:
        subjects_dir = os.environ['SUBJECTS_DIR']
        # extract the subject infromation from the file name
        meg_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        stc_name = name[:name.rfind('-epo.fif')] 
        subject = name.split('_')[0]
        subject_path = subjects_dir + '/%s' %subject
        min_dir = subjects_dir + '/%s' %min_subject
        fn_trans = meg_path + '/%s-trans.fif' % subject
        fn_src = subject_path + '/bem/%s-ico-4-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        # Make sure the target path is exist
        stc_path = min_dir + '/DICS_ROIs/%s' % subject
        set_directory(stc_path)
        # Read the MNI source space
        epochs = mne.read_epochs(fname)
        evoked = epochs.average()
        forward = mne.make_forward_solution(epochs.info, trans=fn_trans,
                                            src=fn_src, bem=fn_bem,
                                            fname=None, meg=True, eeg=False,
                                            mindist=5.0, n_jobs=2,
                                            overwrite=True)
        forward = mne.convert_forward_solution(forward, surf_ori=True)
        from mne.time_frequency import compute_epochs_csd
        from mne.beamformer import dics
        data_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=ctmin, tmax=ctmax, 
                                      fmin=fmin, fmax=fmax)

        noise_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=nctmin, tmax=nctmax,
                                           fmin=fmin, fmax=fmax)
                
        stc = dics(evoked, forward, noise_csd, data_csd)
        from mne import morph_data
        stc_morph = morph_data(subject, min_subject, stc, grade=4, smooth=4)
        stc_morph.save(stc_path + '/%s_%d_%d' % (stc_name, fmin, fmax), ftype='stc')
        if STCs == True:
            stcs_path = stc_path + '/STCs-%s/' %event
            reset_directory(stcs_path)
            stcs = dics(epochs, forward, noise_csd, data_csd)
            s = 0
            while s < len(stcs):
                stc_morph = mne.morph_data(subject, min_subject, stcs[s], grade=4, smooth=4)
                stc_morph.save(stcs_path + '/trial_%s'
                                % (str(s)), ftype='stc')
                s = s + 1

            
def apply_rois(fn_stcs, event='LLst', tmin=0.0, tmax=0.6, tstep=0.05, window=0.2, 
               fmin=4, fmax=8, thr=99, min_subject='fsaverage'): 
    """
    Compute regions of interest (ROI) based on events
    ----------
    fn_stcs : the file name of morphed stc.
    evt: event related with stc
    thr: the percentile of stc's strength
    min_subject: the subject for the common brain space.
    
    """
    from mne import read_source_spaces 
    fnlist = get_files_from_list(fn_stcs)
    # loop across all filenames
    for fn_stc in fnlist:
        name = os.path.basename(fn_stc)
        subject = name.split('_')[0]
        subjects_dir = os.environ['SUBJECTS_DIR']
        min_dir = subjects_dir + '/%s' %min_subject
        labels_path = min_dir + '/DICS_ROIs/%s/%s/' %(subject, event)
        reset_directory(labels_path)
        src = min_dir + '/bem/%s-ico-4-src.fif' %min_subject
        src_inv = read_source_spaces(src)
        stc = mne.read_source_estimate(fn_stc, subject=min_subject) 
        stc = stc.crop(tmin, tmax)
        src_pow = np.sum(stc.data ** 2, axis=1)
        stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
        tbeg = tmin
        while tbeg < tmax:
            tend = tbeg + window
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
                func_label.save(labels_path + '%s_%s_win%.2f_%2f' % (event, str(i), tbeg, tend))
                i = i + 1
            # right hemisphere definition
            j = 0
            while j < len(func_labels_rh):
                func_label = func_labels_rh[j]
                func_label.save(labels_path + '%s_%s_win%2f_%2f' % (event, str(j), tbeg, tend))
                j = j + 1
            tbeg = tbeg + tstep
        
def _cluster_rois(mer_path, label_list):
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
                    label_name = '%s_' % new_pre + \
                        class_label.name.split('_')[-1]
                os.remove(class_list[i])
                os.remove(test_fn)
                fn_newlabel = mer_path + '/%s.label' %label_name
                if os.path.isfile(fn_newlabel):
                    fn_newlabel = fn_newlabel[:fn_newlabel.rfind('-')] + '_new-%s' %fn_newlabel.split('-')[-1]
                mne.write_label(fn_newlabel, com_label)
                print label_name
                class_list[i] = fn_newlabel
                belong = True
            i = i + 1
        if belong is False:
            class_list.append(test_fn)
    return len(class_list)


def merge_rois(labels_path_list, group=False, evelist=['LLst','LLrt']):
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
        if group is False:
            mer_path = labels_path + '/ROIs/'
            reset_directory(mer_path)
            for eve in evelist:
                source_path = labels_path + '/%s' %eve
                for filename in glob.glob(os.path.join(source_path, '*.*')):
                    shutil.copy(filename, mer_path)
        elif group is True:
            mer_path = labels_path + '/merged/'
            reset_directory(mer_path)
            source_path = labels_path + '/standard/' 
            for filename in glob.glob(os.path.join(source_path, '*.*')):
                shutil.copy(filename, mer_path)
        # Merge the individual subject's ROIs
        reducer = True
        while reducer:
            list_dirs = os.walk(mer_path)
            label_list = ['']
            for root, dirs, files in list_dirs:
                for f in files:
                    label_fname = os.path.join(root, f)
                    label_list.append(label_fname)
            label_list = label_list[1:]
            len_class = _cluster_rois(mer_path, label_list)
            if len_class == len(label_list):
                reducer = False  
            
def stan_rois(fname=None, stan_path=None, size=8.0, min_subject='fsaverage'):
    """
    Before merging all ROIs together, the size of ROIs will be standardized.
    Keep every ROIs in a same size
    ----------
    fname: averaged STC of the trials.
    stan_path: path to store all subjects standarlized labels
    size: the radius of every ROI.
    min_subject: the subject for the common brain space. 
    """
    fnlist = get_files_from_list(fname)
    subjects_dir = os.environ['SUBJECTS_DIR']
    # loop across all filenames
    for fn_stc in fnlist:
        stc_path = os.path.split(fn_stc)[0]
        stc_morph = mne.read_source_estimate(fn_stc, subject=min_subject)
        #min_path = subjects_dir + '/%s' %min_subject
        # extract the subject infromation from the file name
        name = os.path.basename(fn_stc)
        subject = name.split('_')[0]
        mer_path = stc_path + '/ROIs/'
        list_dirs = os.walk(mer_path)
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label = mne.read_label(label_fname)
                stc_label = stc_morph.in_label(label)
                src_pow = np.sum(stc_label.data ** 2, axis=1)
                if label.hemi == 'lh':
                    # Get the max MNE value within each ROI
                    seed_vertno = stc_label.vertices[0][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=size, hemis=0,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s_%s' % (subject, f))
                elif label.hemi == 'rh':
                    seed_vertno = stc_label.vertices[1][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=size, hemis=1,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s_%s' % (subject, f))
                    
def group_rois(am_sub=0, com_path=None, mer_path=None):
    """
    choose commont ROIs come out in at least 'sum_sub' subjects
    ----------
    am_sub: the least amount of subjects have the common ROIs.
    com_path: the directory of the common labels.
    mer_path: the directory of the merged rois.
    """
    import shutil
    reset_directory(com_path)
    list_dirs = os.walk(mer_path)
    label_list = ['']
    for root, dirs, files in list_dirs:
        for f in files:
            label_fname = os.path.join(root, f)
            label_list.append(label_fname)
    label_list = label_list[1:]
    for fn_label in label_list:
        fn_name = os.path.basename(fn_label)
        subjects = (fn_name.split('_')[0]).split(',')
        if len(subjects) >= am_sub:
            shutil.copy(fn_label, com_path)
