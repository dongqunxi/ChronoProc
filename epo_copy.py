import os, glob, shutil
event_list = ['LLst', 'LRst', 'RLst', 'RRst']
for event in event_list:
    path = '/home/uais/data/Chrono/mne/*/*/*/*/*evtW_%s_bc-epo.fif' %event
    subjects_dir = os.environ['SUBJECTS_DIR']
    for filename in glob.glob(path):
        name = os.path.basename(filename)
        subject = name.split('_')[0]
        epo_path = subjects_dir + '/%s/MEG/' %subject
        shutil.copy(filename, epo_path)