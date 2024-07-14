import argparse
import os
import os.path as op
import string
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking


def _get_parser():
    parser = argparse.ArgumentParser(description="Run group analysis")
    parser.add_argument(
        "--dset",
        dest="dset",
        required=True,
        help="Path to BIDS directory",
    )
    parser.add_argument(
        "--mriqc_dir",
        dest="mriqc_dir",
        required=True,
        help="Path to MRIQC directory",
    )
    parser.add_argument(
        "--preproc_dir",
        dest="preproc_dir",
        required=True,
        help="Path to fMRIPrep directory",
    )
    parser.add_argument(
        "--clean_dir",
        dest="clean_dir",
        required=True,
        help="Path to denoising directory",
    )
    parser.add_argument(
        "--rsfc_dir",
        dest="rsfc_dir",
        required=True,
        help="Path to RSFC directory",
    )
    parser.add_argument(
        "--sessions",
        dest="sessions",
        default=[None],
        required=False,
        nargs="+",
        help="Sessions identifier, with the ses- prefix.",
    )
    parser.add_argument(
        "--template",
        dest="template",
        default=None,
        required=False,
        help="Template to resample data",
    )
    parser.add_argument(
        "--template_mask",
        dest="template_mask",
        default=None,
        required=False,
        help="Template to resample masks",
    )
    parser.add_argument(
        "--roi_lst",
        dest="roi_lst",
        nargs="+",
        required=True,
        help="ROI label list",
    )
    parser.add_argument(
        "--roi",
        dest="roi",
        required=True,
        help="ROI label",
    )
    parser.add_argument(
        "--program]",
        dest="program",
        required=True,
        help="Program: 3dttest++ or 3dmema",
    )
    parser.add_argument(
        "--n_jobs",
        dest="n_jobs",
        required=True,
        help="CPUs",
    )
    return parser


def conn_resample(roi_in, roi_out, template):

    cmd = f"3dresample \
            -prefix {roi_out} \
            -master {template} \
            -inset {roi_in}"
    print(f"\t\t\t{cmd}", flush=True)
    os.system(cmd)


def remove_ouliers(mriqc_dir, briks_files, mask_files):

    runs_to_exclude_df = pd.read_csv(op.join(mriqc_dir, f"runs_to_exclude.tsv"), sep="\t")
    runs_to_exclude = runs_to_exclude_df["bids_name"].tolist()
    prefixes_tpl = tuple(runs_to_exclude)

    clean_briks_files = [x for x in briks_files if not op.basename(x).startswith(prefixes_tpl)]
    clean_mask_files = [x for x in mask_files if not op.basename(x).startswith(prefixes_tpl)]

    return clean_briks_files, clean_mask_files


def remove_missingdat(participants_df, briks_files, mask_files):
    # print(participants_df, flush=True)
    participants_df = participants_df.dropna()
    # print(participants_df, flush=True)
    subjects_to_keep = participants_df["participant_id"].tolist()

    prefixes_tpl = tuple(subjects_to_keep)

    clean_briks_files = [x for x in briks_files if op.basename(x).startswith(prefixes_tpl)]
    clean_mask_files = [x for x in mask_files if op.basename(x).startswith(prefixes_tpl)]

    return clean_briks_files, clean_mask_files


def subj_ave_roi(clean_subj_dir, subj_briks_files, subjAve_roi_briks_file, roi_idx):
    n_runs = len(subj_briks_files)
    letters = list(string.ascii_lowercase[0:n_runs])

    subj_roi_briks_files = [
        "-{0} {1}'[{2}]'".format(letters[idx], x.split(".HEAD")[0], roi_idx)
        for idx, x in enumerate(subj_briks_files)
    ]
    input_str = " ".join(subj_roi_briks_files)

    # Get weights from number of volumes left in the time series
    weight_lst = []
    for subj_briks_file in subj_briks_files:
        prefix = op.basename(subj_briks_file).split("desc-")[0].rstrip("_")
        censor_files = glob(op.join(clean_subj_dir, f"{prefix}_censoring*.1D"))
        assert len(censor_files) == 1
        censor_file = censor_files[0]
        tr_censor = pd.read_csv(censor_file, header=None)
        tr_left = len(tr_censor.index[tr_censor[0] == 1].tolist())
        weight_lst.append(tr_left)
    # Normalize weights
    weight_norm_lst = [float(x) / sum(weight_lst) for x in weight_lst]

    # Conform equation (a*w[1]+b*w[2]+...)/n_runs
    equation = [f"{letters[idx]}*{round(w,4)}" for idx, w in enumerate(weight_norm_lst)]
    if n_runs > 1:
        equation_str = "+".join(equation)
        exp_str = f"({equation_str})/{n_runs}"
    else:
        exp_str = f"{equation[0]}"

    cmd = f"3dcalc {input_str} -expr '{exp_str}' -prefix {subjAve_roi_briks_file}"
    print(f"\t{cmd}", flush=True)
    os.system(cmd)


def subj_mean_fd(preproc_subj_dir, subj_briks_files, subj_mean_fd_file):
    fd = [
        pd.read_csv(
            op.join(
                preproc_subj_dir,
                "{}_desc-confounds_timeseries.tsv".format(op.basename(x).split("_space-")[0]),
            ),
            sep="\t",
        )["framewise_displacement"].mean()
        for x in subj_briks_files
    ]
    mean_fd = np.mean(fd)
    mean_fd = np.around(mean_fd, 4)
    with open(subj_mean_fd_file, "w") as fo:
        fo.write(f"{mean_fd}")

    return mean_fd


def writearg_1sample(onettest_args_fn, program):
    with open(onettest_args_fn, "w") as fo:
        if program == "3dttest++":
            fo.write("-setA Group\n")
        elif program == "3dmema":
            fo.write("-set Group\n")


def append2arg_1sample(subject, subjAve_roi_coef_file, subjAve_roi_tstat_file, onettest_args_fn, program):
    coef_id = "{coef}'[0]'".format(coef=subjAve_roi_coef_file)
    if program == "3dttest++":
        with open(onettest_args_fn, "a") as fo:
            fo.write(f"{subject} {coef_id}\n")
    elif program == "3dmema":
        tstat_id = "{tstat}'[0]'".format(tstat=subjAve_roi_tstat_file)
        with open(onettest_args_fn, "a") as fo:
            fo.write(f"{subject} {coef_id} {tstat_id}\n")


def get_setAB(subject, subjAve_roi_coef_file, subjAve_roi_tstat_file, participants_df, setA, setB, program):
    sub_df = participants_df[participants_df["participant_id"] == subject]
    coef_id = "{coef}'[0]'".format(coef=subjAve_roi_coef_file)
    if program == "3dttest++":
        if sub_df["group"].values[0] == 1:
            setA.append(f"{subject} {coef_id}\n")
        elif sub_df["group"].values[0] == 2:
            setB.append(f"{subject} {coef_id}\n")
        else:
            pass
    elif program == "3dmema":
        tstat_id = "{tstat}'[0]'".format(tstat=subjAve_roi_tstat_file)
        if sub_df["group"].values[0] == 1:
            setA.append(f"{subject} {coef_id} {tstat_id}\n")
        elif sub_df["group"].values[0] == 2:
            setB.append(f"{subject} {coef_id} {tstat_id}\n")
        else:
            pass
    return setA, setB


def writearg_2sample(setA, setB, twottest_args_fn, program):
    setA = "".join(setA)
    setB = "".join(setB)
    with open(twottest_args_fn, "w") as fo:
        if program == "3dttest++":
            fo.write(f"-setA nonUser\n{setA}-setB User\n{setB}")
        elif program == "3dmema":
            fo.write(f"-set nonUser\n{setA}-set User\n{setB}")


def writecov_1sample(onettest_cov_fn):
    cov_labels = ["subj", "age", "meanfd"]
    # cov_labels = ["subject", "age", "sex", "handedness", "mean_fd"]

    with open(onettest_cov_fn, "w") as fo:
        fo.write("{}\n".format(" ".join(cov_labels)))


def append2cov_1sample(subject, mean_fd, participants_df, onettest_cov_fn):
    # site_dict = {site: i for i, site in enumerate(participants_df["site_id_l"].unique())}
    sub_df = participants_df[participants_df["participant_id"] == subject]
    age = sub_df["age"].values[0]
    # sex = sub_df["sex"].values[0]
    # handedness = sub_df["handedness"].values[0]
    cov_variables = [subject, age, mean_fd]
    # cov_variables = [subject, age, sex, handedness, mean_fd]
    cov_variables_str = [str(x) for x in cov_variables]

    with open(onettest_cov_fn, "a") as fo:
        fo.write("{}\n".format(" ".join(cov_variables_str)))


def run_onesampttest(bucket_fn, mask_fn, covariates_file, args_file, program ,n_jobs):
    with open(args_file) as file:
        arg_list = file.readlines()
    arg_list_up = [x.replace("\n", "") for x in arg_list]
    arg_list = " ".join(arg_list_up)
    if program == "3dttest++":
        cmd = f"3dttest++ -prefix {bucket_fn} \
                -mask {mask_fn} \
                -Covariates {covariates_file} \
                -Clustsim {n_jobs} \
                {arg_list}"
    elif program == "3dmema":
        cmd = f"3dMEMA -prefix {bucket_fn} \
                -mask {mask_fn} \
                -covariates {covariates_file} \
                -verb 1 \
                -jobs {n_jobs} \
                {arg_list}"
    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)


def run_twosampttest(bucket_fn, mask_fn, covariates_file, args_file, program ,n_jobs):
    with open(args_file) as file:
        arg_list = file.readlines()
    arg_list_up = [x.replace("\n", "") for x in arg_list]
    arg_list = " ".join(arg_list_up)
    if program == "3dttest++":
        cmd = f"3dttest++ -prefix {bucket_fn} \
                -BminusA \
                -mask {mask_fn} \
                -Covariates {covariates_file} \
                -Clustsim {n_jobs} \
                {arg_list}"
    elif program == "3dmema":
        cmd = f"3dMEMA -prefix {bucket_fn} \
                -groups nonUser User \
                -mask {mask_fn} \
                -covariates {covariates_file} \
                -verb 1 \
                -jobs {n_jobs} \
                {arg_list}"
    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)


def main(
    dset,
    mriqc_dir,
    preproc_dir,
    clean_dir,
    rsfc_dir,
    sessions,
    template,
    template_mask,
    roi_lst,
    roi,
    program,
    n_jobs,
):
    """Run group analysis workflows on a given dataset."""
    os.system(f"export OMP_NUM_THREADS={n_jobs}")
    roi_coef_dict = {label: x * 3 + 1 for x, label in enumerate(roi_lst)}
    roi_tstat_dict = {label: x * 3 + 2 for x, label in enumerate(roi_lst)}
    print(roi_coef_dict, flush=True)
    space = "MNI152NLin2009cAsym"
    n_jobs = int(n_jobs)
    # Load important tsv files
    participants_df = pd.read_csv(op.join(dset, "participants.tsv"), sep="\t")

    # Define directories
    for session in sessions:
        if session is not None:
            rsfc_subjs_dir = op.join(rsfc_dir, "*", session, "func")
            session_label = f"_{session}"
        else:
            rsfc_subjs_dir = op.join(rsfc_dir, "*", "func")
            session_label = ""

        rsfc_group_dir = op.join(rsfc_dir, f"group-{program}")
        os.makedirs(rsfc_group_dir, exist_ok=True)

        # Collect important files
        briks_files = sorted(
            glob(
                op.join(
                    rsfc_subjs_dir, f"*task-rest*_space-{space}*_desc-norm_bucketREML+tlrc.HEAD"
                )
            )
        )
        mask_files = sorted(
            glob(op.join(rsfc_subjs_dir, f"*task-rest*_space-{space}*_desc-brain_mask.nii.gz"))
        )

        # Remove outliers using MRIQC metrics
        print(f"Number of runs. Briks: {len(briks_files)}, Masks: {len(mask_files)}", flush=True)
        clean_briks_files, clean_mask_files = remove_ouliers(mriqc_dir, briks_files, mask_files)
        print(
            f"Runs after removing outliers data. Briks: {len(clean_briks_files)}, Masks: {len(clean_mask_files)}",
            flush=True,
        )
        clean_briks_files, clean_mask_files = remove_missingdat(
            participants_df, clean_briks_files, clean_mask_files
        )
        print(
            f"Runs after removing missing data. Briks: {len(clean_briks_files)}, Masks: {len(clean_mask_files)}",
            flush=True,
        )
        assert len(clean_briks_files) == len(clean_mask_files)
        # clean_briks_nm = [op.basename(x).split("_space-")[0] for x in clean_briks_files]
        # clean_mask_nm = [op.basename(x).split("_space-")[0] for x in clean_mask_files]
        # clean_briks_tpl = tuple(clean_briks_nm)
        # mask_not_brik = [x for x in clean_mask_nm if not x.startswith(clean_briks_tpl)]

        # Write group file
        clean_briks_fn = op.join(
            rsfc_group_dir, f"sub-group{session_label}_task-rest_space-{space}_briks.txt"
        )
        if not op.exists(clean_briks_fn):
            with open(clean_briks_fn, "w") as fo:
                for tmp_brik_fn in clean_briks_files:
                    fo.write(f"{tmp_brik_fn}\n")

        # Create group mask
        group_mask_fn = op.join(
            rsfc_group_dir,
            f"sub-group{session_label}_task-rest_space-{space}_desc-brain_mask.nii.gz",
        )
        if not op.exists(group_mask_fn):
            if template_mask is None:
                clean_mask_obj = image.load_img(clean_mask_files[0])
                affine, shape = clean_mask_obj.affine, clean_mask_obj.shape
            else:
                template_mask_obj = image.load_img(template_mask)
                affine, shape = template_mask_obj.affine, template_mask_obj.shape
            for clean_mask_file in clean_mask_files:
                clean_mask_img = image.load_img(clean_mask_file)
                if clean_mask_img.shape != shape:
                    clean_res_mask_img = image.resample_img(
                        clean_mask_img, affine, shape[:3], interpolation="nearest"
                    )
                    nib.save(clean_res_mask_img, clean_mask_file)

            group_mask = masking.intersect_masks(clean_mask_files, threshold=0.5)
            nib.save(group_mask, group_mask_fn)

        roi_dir = op.join(rsfc_group_dir, roi)
        os.makedirs(roi_dir, exist_ok=True)
        # Conform onettest_args_fn and twottest_args_fn
        onettest_args_fn = op.join(
            roi_dir, f"sub-group{session_label}_task-rest_desc-1SampletTest{roi}_args.txt"
        )
        twottest_args_fn = op.join(
            roi_dir, f"sub-group{session_label}_task-rest_desc-2SampletTest{roi}_args.txt"
        )
        if not op.exists(onettest_args_fn):
            writearg_1sample(onettest_args_fn, program)

        # Conform onettest_cov_fn and twottest_cov_fn
        onettest_cov_fn = op.join(
            roi_dir, f"sub-group{session_label}_task-rest_desc-1SampletTest{roi}_cov.txt"
        )
        if not op.exists(onettest_cov_fn):
            writecov_1sample(onettest_cov_fn)

        setA = []
        setB = []
        # Calculate subject and ROI level average connectivity
        subjects = [op.basename(x).split("_")[0] for x in clean_briks_files]
        subjects = sorted(list(set(subjects)))
        print(f"Group analysis sample size: {len(subjects)}")

        # Get template
        if template is None:
            for clean_briks_file in clean_briks_files:
                temp_template = op.join(f"{clean_briks_file}'{roi_coef_dict[roi]}'")
                temp_template_img = image.load_img(temp_template)
                if temp_template_img.shape[0] == 81:
                    template = temp_template
                    print(f"Template {template}")
                    break
        else:
            template_img = image.load_img(template)
            template_size = template_img.shape[0:2]
            print(f"Using template {template} with size: {template_size}", flush=True)

        for subject in subjects:
            if session is not None:
                rsfc_subj_dir = op.join(rsfc_dir, subject, session, "func")
                preproc_subj_dir = op.join(preproc_dir, subject, session, "func")
                clean_subj_dir = op.join(clean_dir, subject, session, "func")
            else:
                rsfc_subj_dir = op.join(rsfc_dir, subject, "func")
                preproc_subj_dir = op.join(preproc_dir, subject, "func")
                clean_subj_dir = op.join(clean_dir, subject, "func")
            subj_briks_files = [x for x in clean_briks_files if subject in x]

            if "run-" in subj_briks_files[0]:
                prefix = op.basename(subj_briks_files[0]).split("run-")[0].rstrip("_")
            else:
                prefix = op.basename(subj_briks_files[0]).split("space-")[0].rstrip("_")

            subjAve_roi_coef_file = op.join(
                rsfc_subj_dir,
                f"{prefix}_space-{space}_desc-ave{roi}_coef",
            )
            subjAveRes_roi_coef_file = op.join(
                rsfc_subj_dir,
                f"{prefix}_space-{space}_desc-ave{roi}res_coef",
            )
            subjAve_roi_tstat_file = op.join(
                rsfc_subj_dir,
                f"{prefix}_space-{space}_desc-ave{roi}_tstat",
            )
            subjAveRes_roi_tstat_file = op.join(
                rsfc_subj_dir,
                f"{prefix}_space-{space}_desc-ave{roi}res_tstat",
            )
            subj_mean_fd_file = op.join(
                rsfc_subj_dir,
                f"{prefix}_meanFD.txt",
            )
            if not op.exists(f"{subjAve_roi_coef_file}+tlrc.BRIK"):
                # Average coefficient across runs
                subj_ave_roi(
                    clean_subj_dir, subj_briks_files, subjAve_roi_coef_file, roi_coef_dict[roi]
                )
                # Average tstat across runs
                subj_ave_roi(
                    clean_subj_dir, subj_briks_files, subjAve_roi_tstat_file, roi_tstat_dict[roi]
                )

            # Resample
            subjAve_roi_briks = image.load_img(f"{subjAve_roi_coef_file}+tlrc.BRIK")
            if subjAve_roi_briks.shape[0:2] != template_size:
                if not op.exists(f"{subjAveRes_roi_coef_file}+tlrc.BRIK"):
                    conn_resample(
                        f"{subjAve_roi_coef_file}+tlrc",
                        subjAveRes_roi_coef_file,
                        template,
                    )
                    conn_resample(
                        f"{subjAve_roi_tstat_file}+tlrc",
                        subjAveRes_roi_tstat_file,
                        template,
                    )
                subjAve_roi_coef_file = subjAveRes_roi_coef_file
                subjAve_roi_tstat_file = subjAveRes_roi_tstat_file

            # Get subject level mean FD
            mean_fd = subj_mean_fd(preproc_subj_dir, subj_briks_files, subj_mean_fd_file)

            # Append subject specific info for onettest_args_fn
            if op.exists(onettest_args_fn):
                append2arg_1sample(
                    subject, f"{subjAve_roi_coef_file}+tlrc.BRIK", f"{subjAve_roi_tstat_file}+tlrc.BRIK", onettest_args_fn, program
                )

            # Get setA and setB to write twottest_args_fn
            # if not op.exists(twottest_args_fn):
            setA, setB = get_setAB(
                subject, f"{subjAve_roi_coef_file}+tlrc.BRIK", f"{subjAve_roi_tstat_file}+tlrc.BRIK", participants_df, setA, setB, program
            )

            # Append subject specific info for onettest_cov_fn
            if op.exists(onettest_cov_fn):
                append2cov_1sample(subject, mean_fd, participants_df, onettest_cov_fn)

        # Write twottest_args_fn
        if not op.exists(twottest_args_fn):
            writearg_2sample(setA, setB, twottest_args_fn, program)

        # Statistical analysis
        # Whole-brain, one-sample t-tests
        onettest_briks_fn = op.join(
            roi_dir,
            f"sub-group{session_label}_task-rest_desc-1SampletTest{roi}_briks",
        )
        # Whole-brain, two-sample t-tests
        twottest_briks_fn = op.join(
            roi_dir,
            f"sub-group{session_label}_task-rest_desc-2SampletTest{roi}_briks",
        )

        # Run one-sample ttest
        os.chdir(op.dirname(onettest_briks_fn))
        if not op.exists(f"{onettest_briks_fn}+tlrc.BRIK"):
            run_onesampttest(
                op.basename(onettest_briks_fn),
                group_mask_fn,
                onettest_cov_fn,
                onettest_args_fn,
                program,
                n_jobs,
            )

        # Run two-sample ttest
        os.chdir(op.dirname(twottest_briks_fn))
        if not op.exists(f"{twottest_briks_fn}+tlrc.BRIK"):
            run_twosampttest(
                op.basename(twottest_briks_fn),
                group_mask_fn,
                onettest_cov_fn,
                twottest_args_fn,
                program,
                n_jobs,
            )


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
