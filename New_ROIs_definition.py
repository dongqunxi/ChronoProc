#morphed STCs and average them
from jumeg.jumeg_preprocessing import get_files_from_list
import mne
import os
from dirs_manage import reset_directory, set_directory
def apply_inverse(fnepo, method='dSPM', event='LLst', min_subject='fsaverage', STC_US='ROI', 
                  snr=5.0):
    '''  
        Parameter
        ---------
        fnepo: string or list
            The epochs file with ECG, EOG and environmental noise free.
        method: inverse method, 'MNE' or 'dSPM'
        event: string
            The event name related with epochs.
        min_subject: string
            The subject name as the common brain.
        STC_US: string
            The using of the inversion for further analysis.
            'ROI' stands for ROIs definition, 'CAU' stands for causality analysis.
        snr: signal to noise ratio for inverse solution. 
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import (apply_inverse, apply_inverse_epochs)
    subjects_dir = os.environ['SUBJECTS_DIR']
    fnlist = get_files_from_list(fnepo)
    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        stc_name = name[:name.rfind('-epo.fif')] 
        subject = name.split('_')[0]
        subject_path = subjects_dir + '/%s' %subject
        min_dir = subjects_dir + '/%s' %min_subject
        fn_trans = fn_path + '/%s-trans.fif' % subject
        fn_cov = fn_path + '/%s_empty,nr-cov.fif' % subject
        fn_src = subject_path + '/bem/%s-ico-4-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        snr = snr
        lambda2 = 1.0 / snr ** 2 
        #noise_cov = mne.read_cov(fn_cov)
        epochs = mne.read_epochs(fname)
        noise_cov = mne.read_cov(fn_cov)
        if STC_US == 'ROI':
            # this path used for ROI definition
            stc_path = min_dir + '/%s_ROIs/%s' %(method,subject)
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
            stc_morph = mne.morph_data(subject, subject_id, stc, grade=5)
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
                    subject, min_subject, stcs[s], grade=5)
                stc_morph.save(stcs_path + '/trial%s_fsaverage'
                                % (subject, str(s)), ftype='stc')
                s = s + 1
def apply_STCs(fnstcs, method='dSPM', event='LLst', min_subject='fsaverage', STC_US='ROI', 
               snr=5.0):
    subjects_dir = os.environ['SUBJECTS_DIR']
    fnlist = get_files_from_list(fnstcs)
    # loop across all filenames
    stcs = []
    for fname in fnlist:
        