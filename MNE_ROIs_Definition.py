'''ROIs definition using MNE, noise computing using the pre-stimulus segment
'''
from jumeg.jumeg_preprocessing import get_files_from_list
import mne
import os
import numpy as np
from dirs_manage import reset_directory, set_directory
''' Before running these functions, please make sure essential files in the
    correct pathes as 'bem-sol' and 'src' files: subject_path/bem/. 'trans'
    file is in the same directory as fname_raw.The way of applying these
    functions please refer 'test.py'.
'''

def apply_inverse(fnepo, method='dSPM', event='LLst', min_subject='fsaverage', STC_US='ROI', 
                  condition='LL', save_cov=False):
    '''  
        1. Noise covariance matrix is calculated under the condion of 'require_
        filter'.
        2. 'fnclean' is inversed and morphed into the common source space under 
        the condition of 'require_filter'.
        
        Parameter
        ---------
        fnepo: string or list
            The epochs file with ECG, EOG and environmental noise free.
        method: inverse method, 'MNE' or 'dSPM'
        event: string
            The event name related with epochs.
        min_subject: string
            The subject name as the common brain.
        STC_use: string
            The using of the inversion for further analysis.
            'ROI' stands for ROIs definition, 'CAU' stands for causality analysis. 
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import (apply_inverse, apply_inverse_epochs)
    subjects_dir = os.environ['SUBJECTS_DIR']
    fnlist = get_files_from_list(fnepo)
    # loop across all filenames
    for fname in fnlist:
        meg_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        stc_name = name[:name.rfind('-epo.fif')] 
        subject = name.split('_')[0]
        subject_path = subjects_dir + '/%s' %subject
        min_dir = subjects_dir + '/%s' %min_subject
        fn_trans = meg_path + '/%s-trans.fif' % subject
        fn_src = subject_path + '/bem/%s-ico-4-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        snr = 3.0
        lambda2 = 1.0 / snr ** 2 
        #noise_cov = mne.read_cov(fn_cov)
        epochs = mne.read_epochs(fname)
        fn_cov = meg_path + '/%s-cov.fif' %condition
        if save_cov == False:
            noise_cov = mne.read_cov(fn_cov)
        elif save_cov==True:
            noise_cov = mne.compute_covariance(epochs, tmax=0.)
            mne.write_cov(fn_cov, noise_cov)
        if STC_US == 'ROI':
            # this path used for ROI definition
            stc_path = min_dir + '/MNE_ROIs/%s' % subject
            #fn_cov = meg_path + '/%s_empty,fibp1-45,nr-cov.fif' % subject
            evoked = epochs.average()
            set_directory(stc_path)
            noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                                            mag=0.05, grad=0.05, proj=True)
            fwd_ev = mne.make_forward_solution(evoked.info, trans=fn_trans,
                                                src=fn_src, bem=fn_bem,
                                                fname=None, meg=True, eeg=False,
                                                mindist=5.0, n_jobs=2,
                                                overwrite=True)
            fwd_ev = mne.convert_forward_solution(fwd_ev, surf_ori=True)
            forward_meg_ev = mne.pick_types_forward(fwd_ev, meg=True, eeg=False)
            inverse_operator_ev = mne.minimum_norm.make_inverse_operator(
                evoked.info, forward_meg_ev, noise_cov,
                loose=0.2, depth=0.8)
            # Compute inverse solution
            stc = apply_inverse(evoked, inverse_operator_ev, lambda2, method,
                                pick_ori=None)
            # Morph STC
            subject_id = min_subject
            stc_morph = mne.morph_data(subject, subject_id, stc, grade=4, smooth=4)
            stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
    
        elif STC_US == 'CAU':
            stcs_path = min_dir + '/stcs/%s/%s/' % (subject,event)
            reset_directory(stcs_path)
            noise_cov = mne.cov.regularize(noise_cov, epochs.info,
                                            mag=0.05, grad=0.05, proj=True)
            fwd = mne.make_forward_solution(epochs.info, trans=fn_trans,
                                            src=fn_src, bem=fn_bem,
                                            meg=True, eeg=False, mindist=5.0,
                                            n_jobs=2, overwrite=True)
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)
            forward_meg = mne.pick_types_forward(fwd, meg=True, eeg=False)
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                epochs.info, forward_meg, noise_cov, loose=0.2,
                depth=0.8)
            # Compute inverse solution
            stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                        method=method, pick_ori='normal')
            s = 0
            while s < len(stcs):
                stc_morph = mne.morph_data(
                    subject, min_subject, stcs[s], grade=4, smooth=4)
                stc_morph.save(stcs_path + '/trial%s_fsaverage'
                                % (subject, str(s)), ftype='stc')
                s = s + 1
            
def apply_rois(fn_stc, event, tmin=0.0, tmax=0.3, min_subject='fsaverage', thr=99):
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
    fnlist = get_files_from_list(fn_stc)
    # loop across all filenames
    for ifn_stc in fnlist:
        subjects_dir = os.environ['SUBJECTS_DIR']
        # extract the subject infromation from the file name
        stc_path = os.path.split(ifn_stc)[0]
        #name = os.path.basename(fn_stc)
        #tri = name.split('_')[1].split('-')[0]
        min_path = subjects_dir + '/%s' % min_subject
        fn_src = min_path + '/bem/fsaverage-ico-4-src.fif'
        # Make sure the target path is exist
        labels_path = stc_path + '/%s/' %event
        reset_directory(labels_path)
        # Read the MNI source space
        src_inv = mne.read_source_spaces(fn_src)
        stc = mne.read_source_estimate(ifn_stc, subject=min_subject)
        bc_stc = stc.copy().crop(tmin, tmax)
        src_pow = np.sum(bc_stc.data ** 2, axis=1)
        bc_stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
        #stc_data = stc_morph.data
        #import pdb
        #pdb.set_trace()
        #zscore stc for ROIs estimation
        #d_mu = stc_data.mean(axis=1, keepdims=True)
        #d_std = stc_data.std(axis=1, ddof=1, keepdims=True)
        #z_data = (stc_data - d_mu)/d_std
        func_labels_lh, func_labels_rh = mne.stc_to_label(
            bc_stc, src=src_inv, smooth=True,
            subjects_dir=subjects_dir,
            connected=True)
        # Left hemisphere definition
        i = 0
        while i < len(func_labels_lh):
            func_label = func_labels_lh[i]
            func_label.save(labels_path + '%s_%s' % (event, str(i)))
            i = i + 1
        # right hemisphere definition
        j = 0
        while j < len(func_labels_rh):
            func_label = func_labels_rh[j]
            func_label.save(labels_path + '%s_%s' % (event, str(j)))
            j = j + 1    

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
            mer_path = labels_path + 'merged/'
            reset_directory(mer_path)
            source_path = labels_path + 'standard/' 
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
        #stan_path = min_path + '/Group_ROIs/standard/'
        #set_directory(stan_path)
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