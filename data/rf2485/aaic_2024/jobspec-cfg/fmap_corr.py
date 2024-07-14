import shutil
import os
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix3 as mrt
import subprocess
import sys

### Path handling ###
subject=sys.argv[1]
base_dir = "/gpfs/data/lazarlab/CamCan995"
raw_data = os.path.join(base_dir, "raw")
project_dir = os.path.join(base_dir, "derivatives/fmap_processing/")
out_dir = os.path.join(project_dir, subject)

working_path = os.path.join(out_dir, 'working.mif')
intermediatepath = os.path.join(out_dir, 'intermediate_nifti')
tmp_dir = os.path.join(out_dir, "tmp")
fmap_dir = os.path.join(out_dir, "fmap")
if not os.path.exists(fmap_dir):
    os.makedirs(fmap_dir)

nii_tmp_b0 = os.path.join(tmp_dir, "mean_b0.nii")
undistorted_name = "3_dwi_undistorted"
nii_undistorted = os.path.join(intermediatepath, undistorted_name + '.nii')
bvec_undistorted = os.path.join(intermediatepath, undistorted_name + '.bvec')
bval_undistorted = os.path.join(intermediatepath, undistorted_name + '.bval')
json_undistorted = os.path.join(intermediatepath, undistorted_name + '.json')

mag1_raw = os.path.join(raw_data, subject, "fmap", subject + "_magnitude1.nii.gz")
mag1_init = os.path.join(fmap_dir, "mag1_raw.nii.gz")
shutil.copy2(mag1_raw, mag1_init)
phase_raw = os.path.join(raw_data, subject, "fmap", subject + "_phasediff.nii.gz")
phase_init = os.path.join(fmap_dir, "phase_raw.nii.gz")
shutil.copy2(phase_raw, phase_init)
t1_raw = os.path.join(raw_data, subject, "anat", subject + "_T1w.nii.gz")
t1_init = os.path.join(fmap_dir, "t1_raw.nii.gz")
shutil.copy2(t1_raw, t1_init)

os.chdir(tmp_dir)
### Fmap Preprocessing ###
fmap_step = 0 #count fmap processing step

##mag standard space rigid alignment
fmap_step += 1
mag_std_name = str(fmap_step) + '_mag_std'
nii_mag_std = os.path.join(fmap_dir, mag_std_name + '.nii')
mag_std = fsl.Reorient2Std(in_file=mag1_init, 
                           output_type='NIFTI').run().outputs.out_file
os.rename(mag_std, nii_mag_std)

##mag skull strip
fmap_step += 1
mag_brain_name = str(fmap_step) + '_mag_brain'
nii_mag_brain = os.path.join(fmap_dir, mag_brain_name + '.nii')
nii_mag_brain_mask = os.path.join(fmap_dir, mag_brain_name + '_mask.nii')
subprocess.run("mri_synthstrip -i " + nii_mag_std + " -o " + nii_mag_brain + " -m " + nii_mag_brain_mask,
               shell=True)
nii_mag_brain_eroded = os.path.join(fmap_dir, mag_brain_name + '_eroded.nii')
mag_brain_eroded = fsl.ErodeImage(in_file=nii_mag_brain,
                           output_type='NIFTI').run().outputs.out_file
os.rename(mag_brain_eroded, nii_mag_brain_eroded)

##mag head mask
fmap_step += 1
mag_head_name = str(fmap_step) + '_mag_head'
nii_mag_head_mask = os.path.join(fmap_dir, mag_head_name + '_mask.nii')
mag_head_mask = fsl.DilateImage(in_file=nii_mag_brain_mask,
                                kernel_shape='box',
                                kernel_size=20,
                                output_type='NIFTI',
                                operation='mean').run().outputs.out_file
mag_head_mask = fsl.UnaryMaths(in_file=mag_head_mask,
                               output_type='NIFTI',
                               operation='bin').run().outputs.out_file
os.rename(mag_head_mask, nii_mag_head_mask)
nii_mag_head = os.path.join(fmap_dir, mag_head_name + '.nii')
mag_head = fsl.ApplyMask(in_file=nii_mag_std,
                         mask_file=nii_mag_head_mask).run().outputs.out_file
os.rename(mag_head, nii_mag_head)

##t1 standard space rigid alignment
fmap_step += 1
t1_std_name = str(fmap_step) + '_t1_std'
nii_t1_std = os.path.join(fmap_dir, t1_std_name + '.nii')
t1_std = fsl.Reorient2Std(in_file=t1_init,
                          output_type='NIFTI').run().outputs.out_file
os.rename(t1_std, nii_t1_std)

##t1 bias field correction
fmap_step += 1
t1_biascorr_name = str(fmap_step) + "_t1_biascorr"
nii_t1_biascorr = os.path.join(fmap_dir, t1_biascorr_name + '.nii')
t1_biascorr = ants.N4BiasFieldCorrection(copy_header=True,
                                         input_image=nii_t1_std).run().outputs.output_image
os.rename(t1_biascorr, nii_t1_biascorr)
# subprocess.run("N4BiasFieldCorrection -i " + nii_t1_std + " -o " + nii_t1_biascorr, shell=True)

##t1 skull strip
fmap_step += 1
t1_brain_name = str(fmap_step) + '_t1_brain'
nii_t1_brain = os.path.join(fmap_dir, t1_brain_name + '.nii')
subprocess.run("mri_synthstrip -i " + nii_t1_biascorr + " -o " + nii_t1_brain,
               shell=True)

##phase standard space rigid alignment
fmap_step += 1
phase_std_name = str(fmap_step) + '_phase_std'
nii_phase_std = os.path.join(fmap_dir, phase_std_name + '.nii')
phase_std = fsl.Reorient2Std(in_file=phase_init,
                             output_type='NIFTI').run().outputs.out_file
os.rename(phase_std, nii_phase_std)

##generate fieldmap
fmap_step += 1
fmap_name = str(fmap_step) + '_fmap'
nii_fmap = os.path.join(fmap_dir, fmap_name + '.nii')
fmap = fsl.PrepareFieldmap(in_magnitude=nii_mag_brain_eroded,
                           output_type='NIFTI',
                           in_phase=nii_phase_std).run().outputs.out_fieldmap
os.rename(fmap, nii_fmap)

##generate voxel shift map and register fmap to B0
fmap_step += 1
vsm_name = str(fmap_step) + '_vsm'
nii_vsm = os.path.join(fmap_dir, vsm_name + '.nii')
fmap2b0_name = str(fmap_step) + 'fmap2b0'
mat_fmap2b0 = os.path.join(fmap_dir, fmap2b0_name + '.mat')
epi_reg_outputs = fsl.EpiReg(epi=nii_tmp_b0,
                             t1_brain=nii_t1_brain,
                             t1_head=nii_t1_biascorr,
                             echospacing=0.000360002,
                             fmap=nii_fmap,
                             fmapmag=nii_mag_head,
                             fmapmagbrain=nii_mag_brain_eroded,
                             pedir='-y').run().outputs
vsm = epi_reg_outputs.shiftmap
os.rename(vsm, nii_vsm)
fmap2b0 = epi_reg_outputs.fmap2epi_mat
os.rename(fmap2b0, mat_fmap2b0)

##register head mask to B0
fmap_step += 1
b0_head_name = str(fmap_step) + '_b0_head'
nii_b0_head_mask = os.path.join(fmap_dir, b0_head_name + '_mask.nii')
b0_head_mask = fsl.FLIRT(in_file=nii_mag_head_mask,
                         apply_xfm=True,
                         in_matrix_file=mat_fmap2b0,
                         padding_size=0,
                         interp='trilinear',
                         output_type='NIFTI',
                         reference=nii_tmp_b0).run().outputs.out_file
os.rename(b0_head_mask, nii_b0_head_mask)

### Unwarp DWI ###
step_count = 4
dwi_unwarp_name = str(step_count) + '_dwi_unwarp'
nii_dwi_unwarp = os.path.join(intermediatepath, dwi_unwarp_name + '.nii')
dwi_unwarp = fsl.FUGUE(in_file=nii_undistorted,
                       shift_in_file=nii_vsm,
                       mask_file=nii_b0_head_mask,
                       output_type='NIFTI').run().outputs.unwarped_file
os.rename(dwi_unwarp, nii_dwi_unwarp)
bvec_dwi_unwarp = os.path.join(intermediatepath, dwi_unwarp_name + '.bvec')
bval_dwi_unwarp = os.path.join(intermediatepath, dwi_unwarp_name + '.bval')
json_dwi_unwarp = os.path.join(intermediatepath, dwi_unwarp_name + '.json')
shutil.copy2(bvec_undistorted, bvec_dwi_unwarp)
shutil.copy2(bval_undistorted, bval_dwi_unwarp)
shutil.copy2(json_undistorted, json_dwi_unwarp)
mif_dwi_unwarp = mrt.MRConvert(in_file=nii_dwi_unwarp,
                         in_bval=bval_dwi_unwarp,
                         in_bvec=bvec_dwi_unwarp,
                         json_import=json_dwi_unwarp).run().outputs.out_file
os.remove(working_path)
os.rename(mif_dwi_unwarp, working_path)
shutil.rmtree(tmp_dir)