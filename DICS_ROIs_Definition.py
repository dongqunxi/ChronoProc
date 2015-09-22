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
def apply_create_noise_covariance(fname_empty_room, verbose=None):
    
    '''
    Creates the noise covariance matrix from an empty room file.

    Parameters
    ----------
    fname_empty_room : String containing the filename
        of the empty room file (must be a fif-file)
    require_filter: bool
        If true, the empy room file is filtered before calculating
        the covariance matrix. (Beware, filter settings are fixed.)
    verbose : bool, str, int, or None
        If not None, override default verbose level
        (see mne.verbose).
        default: verbose=None
    '''

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from mne import compute_raw_data_covariance as cp_covariance
    from mne import write_cov, pick_types
    from mne.io import Raw
    from jumeg.jumeg_noise_reducer import noise_reducer
    fner = get_files_from_list(fname_empty_room)
    nfiles = len(fner)
    ext_empty_raw = '-raw.fif'
    ext_empty_cov = '-cov.fif'
    # loop across all filenames
    for ifile in range(nfiles):
        fn_in = fner[ifile]
        print ">>> create noise covariance using file: "
        path_in, name = os.path.split(fn_in)
        print name   
        fn_empty_nr = fn_in[:fn_in.rfind('-raw.fif')] + ',nr-raw.fif'
        noise_reducer(fn_in, refnotch=50, detrending=False, fnout=fn_empty_nr)
        noise_reducer(fn_empty_nr, refnotch=60, detrending=False, fnout=fn_empty_nr) 
        noise_reducer(fn_empty_nr, reflp=5, fnout=fn_empty_nr)
        # file name for saving noise_cov
        fn_out = fn_empty_nr[:fn_empty_nr.rfind(ext_empty_raw)] + ext_empty_cov
        # read in data
        raw_empty = Raw(fn_empty_nr, verbose=verbose)
        # pick MEG channels only
        picks = pick_types(raw_empty.info, meg=True, ref_meg=False, eeg=False,
                           stim=False, eog=False, exclude='bads')

        # calculate noise-covariance matrix
        noise_cov_mat = cp_covariance(raw_empty, picks=picks, verbose=verbose)

        # write noise-covariance matrix to disk
        write_cov(fn_out, noise_cov_mat)
   
def apply_inverse(fn_epo, event_id=1,ctmin=0.05, ctmax=0.25, fmin=4, fmax=8, 
                  min_subject='fsaverage', save_forward=False):
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
    save_forward: Whether save the forward solution or not.
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
        tmin = epochs.times.min()
        tmax = epochs.times.max()
        fn_empty = meg_path + '/%s_empty,nr-raw.fif' % subject
        raw_noise = Raw(fn_empty, preload=True)
        epochs.info['bads'] = raw_noise.info['bads']
        picks_noise = pick_types(raw_noise.info, meg='mag', exclude='bads')
        events_noise = make_fixed_length_events(raw_noise, event_id, duration=1.)
        epochs_noise = Epochs(raw_noise, events_noise, event_id, tmin,
                                tmax, proj=True, picks=picks_noise,
                                baseline=None, preload=True, reject=None)
        # Make sure the number of noise epochs is the same as data epochs
        epochs_noise = epochs_noise[:len(epochs.events)]
        evoked = epochs.average()
        forward = mne.make_forward_solution(epochs.info, trans=fn_trans,
                                            src=fn_src, bem=fn_bem,
                                            fname=None, meg=True, eeg=False,
                                            mindist=5.0, n_jobs=2,
                                            overwrite=True)
        forward = mne.convert_forward_solution(forward, surf_ori=True)
        if save_forward == True:
            fn_fwd = fname[:fname.rfind('-epo.fif')] + '-fwd.fif'
            mne.write_forward_solution(fn_fwd, forward, overwrite=True)
        from mne.time_frequency import compute_epochs_csd
        from mne.beamformer import dics
        data_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=ctmin, tmax=ctmax, 
                                      fmin=fmin, fmax=fmax)

        noise_csd = compute_epochs_csd(epochs_noise, mode='multitaper', tmin=ctmin, tmax=ctmax,
                                           fmin=fmin, fmax=fmax)
                
        stc = dics(evoked, forward, noise_csd, data_csd)
        from mne import morph_data
        stc_morph = morph_data(subject, min_subject, stc, grade=4, smooth=4)
        stc_morph.save(stc_path + '/%s_%d_%d' % (stc_name, fmin, fmax), ftype='stc')
     
def apply_rois(fn_stcs, evt='LLst', tmin=0.05, tmax=0.25, thr=99, min_subject='fsaverage'): 
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
        labels_path = min_dir + '/DICS_ROIs/%s/%s/' %(subject, evt)
        reset_directory(labels_path)
        src = min_dir + '/bem/%s-ico-4-src.fif' %min_subject
        src_inv = read_source_spaces(src)
        stc = mne.read_source_estimate(fn_stc, subject=min_subject) 
        stc = stc.crop(tmin, tmax)
        src_pow = np.sum(stc.data ** 2, axis=1)
        stc.data[src_pow < np.percentile(src_pow, thr)] = 0.
        #stc_data = stc_morph.data
        #import pdb
        #pdb.set_trace()
        #zscore stc for ROIs estimation
        #d_mu = stc_data.mean(axis=1, keepdims=True)
        #d_std = stc_data.std(axis=1, ddof=1, keepdims=True)
        #z_data = (stc_data - d_mu)/d_std
        func_labels_lh, func_labels_rh = mne.stc_to_label(
            stc, src=src_inv, smooth=True,
            subjects_dir=subjects_dir,
            connected=True)
        # Left hemisphere definition
        i = 0
        while i < len(func_labels_lh):
            func_label = func_labels_lh[i]
            func_label.save(labels_path + '%s_%s' % (evt, str(i)))
            i = i + 1
        # right hemisphere definition
        j = 0
        while j < len(func_labels_rh):
            func_label = func_labels_rh[j]
            func_label.save(labels_path + '%s_%s' % (evt, str(j)))
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
            
def stan_rois(fname, size=8.0, min_subject='fsaverage'):
    """
    Before merging all ROIs together, the size of ROIs will be standardized.
    Keep every ROIs in a same size
    ----------
    fname: averaged STC of the trials.
    size: the radius of every ROI.
    min_subject: the subject for the common brain space.
    """
    fnlist = get_files_from_list(fname)
    subjects_dir = os.environ['SUBJECTS_DIR']
    # loop across all filenames
    for fn_stc in fnlist:
        stc_path = os.path.split(fn_stc)[0]
        stc_morph = mne.read_source_estimate(fn_stc, subject=min_subject)
        min_path = subjects_dir + '/%s' %min_subject
        # extract the subject infromation from the file name
        name = os.path.basename(fn_stc)
        subject = name.split('_')[0]
        mer_path = stc_path + '/ROIs/'
        stan_path = min_path + '/Group_ROIs/standard/'
        set_directory(stan_path)
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
                    
def group_rois(am_sub=0, min_subject='fsaverage'):
    """
    choose commont ROIs come out in at least 'sum_sub' subjects
    ----------
    am_sub: the least amount of subjects have the common ROIs.
    subjects_dir: the directory of the subjects.
    min_subject: the subject for the common brain space.
    """
    import shutil
    subjects_dir = os.environ['SUBJECTS_DIR']
    min_path = subjects_dir + '/%s' %min_subject 
    com_path = min_path + '/Group_ROIs/common/'
    mer_path = min_path + '/Group_ROIs/merged/'
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
