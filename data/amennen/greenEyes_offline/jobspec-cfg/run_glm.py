#!/usr/bin/env python

# Run from code/ with something like
# ./story_glm.py 001 |& tee ../data/derivatives/logs/glm_log.txt &

from sys import argv
from os import chdir, makedirs
from os.path import exists, join
from subprocess import call
import numpy as np

subject = argv[1]
task = 'prettymouth'

base_dir = f'/jukebox/hasson/snastase/narratives/{task}'
scripts_dir = join(base_dir, 'code')
func_dir = join(base_dir, 'sub-'+subject, 'func')
deriv_dir = join(base_dir, 'derivatives')

# Convert fmriprep's confounds.tsv for 3dDeconvolve -ortvec per run
prep_dir = join(deriv_dir, 'fmriprep', 'sub-'+subject, 'func')
glm_dir = join(deriv_dir, 'afni', 'sub-'+subject, 'func')
reg_dir = join(glm_dir, 'regressors')
if not exists(reg_dir):
    makedirs(reg_dir)

with open(join(prep_dir,
               'sub-{0}_task-{1}_desc-confounds_regressors.tsv'.format(
                subject, task))) as f:
    lines = [line.strip().split('\t') for line in f.readlines()]

confounds = {}
for confound_i, confound in enumerate(lines[0]):
    confound_ts = []
    for tr in lines[1:]:
        confound_ts.append(tr[confound_i])
    confounds[confound] = confound_ts

keep = ['framewise_displacement', 'a_comp_cor_00', 'a_comp_cor_01',
        'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04',
        'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z']
ortvec = {c: confounds[c] for c in keep}

# Create de-meaned and first derivatives of head motion (backward difference with leading zero)
for motion_reg in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']:
    motion = [float(m) for m in ortvec[motion_reg]]
    motion_demean = np.array(motion) - np.mean(motion)
    motion_deriv = np.insert(np.diff(motion_demean, n=1), 0, 0)
    assert len(motion_demean) == len(motion_deriv) == len(ortvec[motion_reg])
    ortvec[motion_reg + '_demean'] = ['{:.9f}'.format(m) for m in motion_demean]
    ortvec[motion_reg + '_deriv'] = ['{:.9f}'.format(m) for m in motion_deriv]
    del ortvec[motion_reg]

assert len(ortvec) == 19

with open(join(reg_dir, 'sub-{0}_task-{1}_desc-ortvec_regressors.1D'.format(subject, task)), 'w') as f:
    rows = []
    for tr in range(len(ortvec['framewise_displacement'])):
        row = []
        for confound in ortvec:
            if ortvec[confound][tr] == 'n/a':
                row.append('0')
            else:
                row.append(ortvec[confound][tr])
        assert len(row) == 19
        row = '\t'.join(row)
        rows.append(row)
    #assert len(rows) == n_vol
    f.write('\n'.join(rows))

# Change directory for AFNI
chdir(glm_dir)

# Run AFNI's 3dTproject (sub-084_task-pieman_space-fsaverage6_hemi-L.func.gii)
# HaxbyLab filter "-passband 0.00667 0.1", Hasson Lab "-stopband 0 0.00714
space = 'fsaverage6'
for side, hemi in [('L', 'lh'), ('R', 'rh')]:
    cmd = ("3dTproject -polort 2 -stopband 0 0.00714 -TR 1.5 -input " 
                "{0}/sub-{1}_task-{2}_space-fsaverage6_hemi-{3}.func.gii "
                "-ort {4} "
                "-prefix {5}/sub-{1}_task-{2}_space-fsaverage6_hemi-{6}.tproject.gii".format(
                    prep_dir, subject, task, side,
                    join(reg_dir, 'sub-{0}_task-{1}_desc-ortvec_regressors.1D'.format(subject, task)),
                    glm_dir, hemi))
    print("Regressing out confounds using AFNI's 3dTproject"
          "\tSubject {0}, {1}, {2} hemisphere".format(subject, task, hemi))
    call(cmd, shell=True)
    print("\tFinished regressing out confounds!")

# Run AFNI's 3dTproject
space = 'MNI152NLin2009cAsym' 
cmd = ("3dTproject -polort 2 -stopband 0 0.00714 -TR 1.5 -input " 
            "{0}/sub-{1}_task-{2}_space-{3}_desc-preproc_bold.nii.gz "
            "-ort {4} -mask {0}/sub-{1}_task-{2}_space-{3}_desc-brain_mask.nii.gz "
            "-prefix {5}/sub-{1}_task-{2}_space-{3}_desc-tproject_bold.nii.gz".format(
                prep_dir, subject, task, space,
                join(reg_dir, 'sub-{0}_task-{1}_desc-ortvec_regressors.1D'.format(subject, task)),
                glm_dir))
print("Regressing out confounds using AFNI's 3dTproject"
      "\tSubject {0}, {1}".format(subject, task))
call(cmd, shell=True)
print("\tFinished regressing out confounds!")
