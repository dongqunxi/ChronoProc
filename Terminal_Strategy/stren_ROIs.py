from jumeg.jumeg_preprocessing import get_files_from_list
import mne
import os
import numpy as np
from dirs_manage import reset_directory, set_directory
''' We normalized the STCs, to improve th SNR, then average them into a common STCs
'''
subjects_dir = os.environ['SUBJECTS_DIR']
def apply_inverse(fnevo, method='dSPM', snr=5.0, event='LLst', 
                  baseline=False, btmin=-0.3, btmax=-0.1, min_subject='fsaverage'):
    '''  
        Parameter
        ---------
        fnevo: string or list
            The evoked file with ECG, EOG and environmental noise free.
        method: inverse method, 'MNE' or 'dSPM'
        event: string
            The event name related with epochs.
        min_subject: string
            The subject name as the common brain.
        snr: signal to noise ratio for inverse solution. 
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import apply_inverse
    fnlist = get_files_from_list(fnevo)
    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        stc_name = name[:name.rfind('-ave.fif')] 
        subject = name.split('_')[0]
        subject_path = subjects_dir + '/%s' %subject
        min_dir = subjects_dir + '/%s' %min_subject
        fn_trans = fn_path + '/%s-trans.fif' % subject
        fn_cov = fn_path + '/%s_empty,nr-cov.fif' % subject
        fn_src = subject_path + '/bem/%s-ico-5-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        snr = snr
        lambda2 = 1.0 / snr ** 2 
        #noise_cov = mne.read_cov(fn_cov)
        [evoked] = mne.read_evokeds(fname)
        noise_cov = mne.read_cov(fn_cov)
        # this path used for ROI definition
        stc_path = min_dir + '/%s_ROIs/%s' %(method,subject)
        #fn_cov = meg_path + '/%s_empty,fibp1-45,nr-cov.fif' % subject
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
        stc_morph = mne.morph_data(subject, min_subject, stc, grade=5, smooth=5)
        stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
        if baseline == True:
            stc_base = stc_morph.crop(btmin, btmax)
            stc_base.save(stc_path + '/%s_%s_baseline' % (subject, event), ftype='stc')

def apply_norm(fn_stc, event, ref_event, thr=95):        
    fn_list = get_files_from_list(fn_stc)
    stcs = []
    for fname in fn_list:
        fn_path = os.path.split(fname)[0]
        stc = mne.read_source_estimate(fname)       
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        fn_base = fn_path + '/%s_%s_baseline-lh.stc' %(subject,ref_event)
        stc = mne.read_source_estimate(fname)
        stc_base = mne.read_source_estimate(fn_base)
        thre = np.percentile(stc_base.data, thr, axis=-1)
        data = stc.data
        cal_mean = data.mean(axis=-1)
        norm_data = (data.T / thre) - 1
        norm_data[norm_data < 0] = 0
        norm_data = norm_data.T
        norm_data[cal_mean == 0, :] = 0
        norm_mean = norm_data.mean(axis=-1)
        zc_data = norm_data.T/norm_data.max(axis=-1)
        zc_data = zc_data.T
        zc_data[norm_mean == 0, :] = 0
        #import pdb
        #pdb.set_trace()
        print zc_data.min()
        stc.data.setfield(zc_data, np.float32)
        stcs.append(stc)
        fn_nr = fname[:fname.rfind('-lh')] + '_norm_1'
        stc.save(fn_nr, ftype='stc')
    stcs_path = subjects_dir+'/fsaverage/dSPM_ROIs/' 
    fn_avg = stcs_path + '%s' %(event)
    stcs = np.array(stcs)
    stc_avg = np.sum(stcs, axis=0)/stcs.shape[0]
    stc_avg.save(fn_avg, ftype='stc')

def apply_rois(fn_stc_list, event, min_subject='fsaverage', k=15, radius=20., tmin=0.0, tmax=0.3, thr=95):
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
    #from scipy.signal import detrend
    #from scipy.stats.mstats import zscore 
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    fnlist = get_files_from_list(fn_stc_list)
    # loop across all filenames
    for fn_stc in fnlist:
        # extract the subject infromation from the file name
        stc_path = os.path.split(fn_stc)[0]
        min_path = subjects_dir + '/%s' % min_subject
        fn_src = min_path + '/bem/fsaverage-ico-5-src.fif'
        # Make sure the target path is exist
        labels_path = stc_path + '/%s/ini' %event
        reset_directory(labels_path)
      
        # Read the MNI source space
        stc = mne.read_source_estimate(fn_stc)
        stc_intr = stc.copy().crop(tmin, tmax)
        src_pow = np.sum(stc_intr.data, axis=1)
        stc_intr.data[src_pow < np.percentile(src_pow, thr)] = 0.
        cal_mean=stc_intr.data.mean(axis=-1)
        non_data=stc_intr.data[cal_mean>0]
        inds = np.argwhere(cal_mean)
        pca = PCA(n_components=k).fit(non_data)
        kmeans = KMeans(init=pca.components_, n_clusters=8)
        kmeans.fit(non_data)
        codebook = kmeans.cluster_centers_
        #book = np.array((whitened[1], 5)
        #codebook, distortion = kmeans(whitened,9, iter=1000)
        centroids = []
        c_ins = []
        centroid = non_data[0]
        for cb in codebook:
            min_diff =  np.abs(np.linalg.norm(cb) - np.linalg.norm(non_data[0]))
            c_in = 0
            i = 1
            while i < len(non_data):
                wt = non_data[i]
                diff =  np.abs(np.linalg.norm(cb) - np.linalg.norm(wt))
                if min_diff > diff:
                    centroid = wt
                    min_diff = diff
                    c_in = i
                i = i + 1
            centroids.append(centroid)
            c_ins.append(c_in)
            
        centroids = np.array(centroids)    
        c_ins = np.array(c_ins)        
        c_stc=inds[c_ins]
        seeds = np.squeeze(c_stc)
        non_index_lh = seeds[seeds < 10242]
        if non_index_lh.shape != []:    
            func_labels_lh = mne.grow_labels(min_subject, non_index_lh,
                                            extents=radius, hemis=0, 
                                            subjects_dir=subjects_dir, n_jobs=1)
            i = 0
            while i < len(func_labels_lh):
                func_label = func_labels_lh[i]
                func_label.save(labels_path + '/%s_%d' %(event, i))
                i = i + 1
                
        seeds_rh = seeds - 10242
        non_index_rh = seeds_rh[seeds_rh > 0]
        if non_index_rh.shape != []:
            func_labels_rh = mne.grow_labels(min_subject, non_index_rh,
                                            extents=radius, hemis=1,
                                            subjects_dir=subjects_dir, n_jobs=1)                                             
    
            # right hemisphere definition
            j = 0
            while j < len(func_labels_rh):
                func_label = func_labels_rh[j]
                func_label.save(labels_path + '/%s_%d' %(event, j))
                j = j + 1

def _cluster1_rois(mer_path, label_list):
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
                    label_name = '%s_' % (new_pre) + \
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

def apply_merge(labels_path, evt_list):
    import glob, shutil
    mer_path = labels_path + 'ROIs/merge/'
    reset_directory(mer_path)
    source = []
    for evt in evt_list:
        source_path = labels_path + '/%s/ini/' %evt
        source = source + glob.glob(os.path.join(source_path, '*.*'))
    for filename in source:
        shutil.copy(filename, mer_path)
    reducer = True
    while reducer:
        list_dirs = os.walk(mer_path)
        label_list = ['']
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label_list.append(label_fname)
        label_list = label_list[1:]
        len_class = _cluster1_rois(mer_path, label_list)
        if len_class == len(label_list):
            reducer = False  

def apply_comSTC(stcs_path, evt_list):
    import glob
    stcs = []
    for evt in evt_list:
        fname = stcs_path + '%s-lh.stc' %evt
        stc = mne.read_source_estimate(fname)
        stcs.append(stc) 
    stcs = np.array(stcs)
    stc_avg = np.sum(stcs, axis=0)/stcs.shape[0]
    fn_avg = subjects_dir+'/fsaverage/dSPM_ROIs/common' 
    stc_avg.save(fn_avg, ftype='stc')

def apply_stand(fn_stc, radius=10.0, min_subject='fsaverage'):

    """
    ----------
    fname: averaged STC of the trials.
    radius: the radius of every ROI. 
    """
    fnlist = get_files_from_list(fn_stc)
    # loop across all filenames
    for fn_stc in fnlist:
        import glob, shutil
        labels_path = os.path.split(fn_stc)[0]
        stan_path = labels_path + '/ROIs/stand/' 
        reset_directory(stan_path)
        source_path = labels_path + '/ROIs/merge/' 
        source = glob.glob(os.path.join(source_path, '*.*'))
        for filename in source:
            shutil.copy(filename, stan_path)
        stc = mne.read_source_estimate(fn_stc, subject=min_subject)
        list_dirs = os.walk(stan_path)
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label = mne.read_label(label_fname)
                stc_label = stc.in_label(label)
                src_pow = np.sum(stc_label.data ** 2, axis=1)
                if label.hemi == 'lh':
                    # Get the max MNE value within each ROI
                    seed_vertno = stc_label.vertices[0][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=radius, hemis=0,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s' %f)
                elif label.hemi == 'rh':
                    seed_vertno = stc_label.vertices[1][np.argmax(src_pow)]
                    func_label = mne.grow_labels(min_subject, seed_vertno,
                                                 extents=radius, hemis=1,
                                                 subjects_dir=subjects_dir,
                                                 n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(stan_path + '%s' %f)
def _sortlist(label_list, stc, src):
    sort_list = []
    sort_list.append(label_list[0])
    for test_fn in label_list[1:]:
        test_label = mne.read_label(test_fn)
        i = 0
        insert = False
        while (i < len(sort_list)) and insert == False:
            class_label = mne.read_label(sort_list[i])
            class_pca = stc.extract_label_time_course(class_label, src, mode='pca_flip')
            test_pca = stc.extract_label_time_course(test_label, src, mode='pca_flip')
            class_pca = np.squeeze(class_pca)
            test_pca = np.squeeze(test_pca)
            class_pow = np.sum(class_pca ** 2)
            test_pow = np.sum(test_pca ** 2)
            # sort the list
            if test_pow < class_pow:
                sort_list.insert(i, test_fn)
                insert = True
            i = i + 1
             
        if insert == False:
            sort_list.append(test_fn)
       
    return sort_list
    
def _cluster_rois(sel_path, label_list, stc, src, min_dist, mni_subject='fsaverage'):
    """
    subfunctions of merge_ROIs
    ----------
    mer_path: str
        The directory for storing merged ROIs.
    label_list: list
        Labels to be merged
    """
    class_list = []
    label_list = _sortlist(label_list, stc, src)
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
                # Get the centroids
                class_stc = stc.in_label(class_label)
                test_stc = stc.in_label(test_label)
                class_pca = stc.extract_label_time_course(class_label, src, mode='pca_flip')
                test_pca = stc.extract_label_time_course(test_label, src, mode='pca_flip')
                class_pca = np.squeeze(class_pca)
                test_pca = np.squeeze(test_pca)
                class_pow = np.sum(class_pca ** 2)
                test_pow = np.sum(test_pca ** 2)
                
                if class_label.hemi == 'lh':
                    h = 0
                elif class_label.hemi == 'rh':
                    h = 1
                class_seed, _, _ = class_stc.center_of_mass(mni_subject, hemi=h)
                test_seed, _, _ = test_stc.center_of_mass(mni_subject, hemi=h)
                class_mni = mne.vertex_to_mni(class_seed, h, mni_subject)[0]
                test_mni = mne.vertex_to_mni(test_seed, h, mni_subject)[0]
                
                exch = False
                if np.max(class_pow) < np.max(test_pow):
                    exch = True
                
                # Compute the centroids distance   
                if np.linalg.norm(class_mni - test_mni) < min_dist:
                    if exch == True:
                        os.remove(class_list[i])
                        class_list[i] = test_fn
                        class_list = _sortlist(class_list, stc, src)
                    elif exch == False:
                        os.remove(test_fn)
                    belong = True
                i = i + 1
                
        if belong is False:
            class_list.append(test_fn)
            class_list = _sortlist(class_list, stc, src)
                
    return len(class_list)


def apply_sele(fn_stc_list, fn_src, min_dist):
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
    fn_stc_list = get_files_from_list(fn_stc_list)
    # loop across all filenames
    for fn_stc in fn_stc_list:
        import glob, shutil
        labels_path = os.path.split(fn_stc)[0]
        sel_path = labels_path + '/ROIs/sele/' 
        reset_directory(sel_path)
        source_path = labels_path + '/ROIs/stand/' 
        source = glob.glob(os.path.join(source_path, '*.*'))
        for filename in source:
            shutil.copy(filename, sel_path) 
        reducer = True
        stc = mne.read_source_estimate(fn_stc)
        stc.data[stc.data < 0] = 0
        #stc = stc.crop(tmin, tmax)
        src = mne.read_source_spaces(fn_src)
        while reducer:
            list_dirs = os.walk(sel_path)
            label_list = ['']
            for root, dirs, files in list_dirs:
                for f in files:
                    label_fname = os.path.join(root, f)
                    label_list.append(label_fname)
            label_list = label_list[1:]
            len_class = _cluster_rois(sel_path, label_list, stc, src, min_dist)
            if len_class == len(label_list):
                reducer = False  