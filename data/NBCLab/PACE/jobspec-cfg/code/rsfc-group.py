import argparse
import os
import os.path as op
import string
from glob import glob

import nibabel as nib
import numpy as np
from neuroCombat import neuroCombat
from nilearn.maskers import NiftiMasker
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
        "--deriv_dir",
        dest="deriv_dir",
        required=True,
        help="Path to derivatives directory",
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


def conn_harmonize(combat_table_fn, group_mask_fn):
    masker = NiftiMasker()
    masker.fit(group_mask_fn)

    data_df = pd.read_csv(combat_table_fn)
    in_imgs = data_df["input"].to_list()
    out_imgs = data_df["output"].to_list()

    img_lst = []
    img_affines = []
    img_headers = []
    for img in in_imgs:
        img_nii = nib.load(img)
        img_lst.append(masker.transform(img_nii))

        img_affines.append(img_nii.affine)
        img_headers.append(img_nii.header)

    data = np.vstack(img_lst).T
    covars = data_df[["site", "group"]]

    categorical_cols = ["group"]
    batch_col = "site"

    assert data.shape[1] == covars.shape[0]

    model_combat = neuroCombat(
        dat=data, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols
    )
    data_combat = model_combat["data"]

    for i_img, out_img in enumerate(out_imgs):
        new_map = data_combat[:, i_img]
        new_map_nii = masker.inverse_transform(new_map).get_fdata()
        new_map_nii = new_map_nii[..., np.newaxis]

        new_img = nib.Nifti1Image(new_map_nii, img_affines[i_img], img_headers[i_img])
        new_img.to_filename(f"{out_img}.nii")

        cmd = f"3dcopy \
                {out_img}.nii \
                {out_img}"

        if op.exists(f"{out_img}+tlrc.BRIK"):
            os.remove(f"{out_img}+tlrc.BRIK")
        if op.exists(f"{out_img}+tlrc.HEAD"):
            os.remove(f"{out_img}+tlrc.HEAD")

        print(f"\t\t{cmd}", flush=True)
        os.system(cmd)


def remove_ouliers(mriqc_dir, briks_files, mask_files):

    runs_to_exclude_df = pd.read_csv(op.join(mriqc_dir, "runs_to_exclude.tsv"), sep="\t")
    runs_to_exclude = runs_to_exclude_df["bids_name"].tolist()
    prefixes_tpl = tuple(runs_to_exclude)

    clean_briks_files = [x for x in briks_files if not op.basename(x).startswith(prefixes_tpl)]
    clean_mask_files = [x for x in mask_files if not op.basename(x).startswith(prefixes_tpl)]

    return clean_briks_files, clean_mask_files


def remove_missingdat(participants_df, briks_files, mask_files):
    if "session" in participants_df.columns:
        participants_df = participants_df.drop(columns=["session"])

    participants_df = participants_df.dropna()
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
        if (program == "3dttest++") or (program == "combat"):
            fo.write("-setA Group\n")
        elif program == "3dmema":
            fo.write("-set Group\n")


def append2arg_1sample(
    subject, subjAve_roi_coef_file, subjAve_roi_tstat_file, onettest_args_fn, program
):
    coef_id = "{coef}'[0]'".format(coef=subjAve_roi_coef_file)
    if (program == "3dttest++") or (program == "combat"):
        with open(onettest_args_fn, "a") as fo:
            fo.write(f"{subject} {coef_id}\n")
    elif program == "3dmema":
        tstat_id = "{tstat}'[0]'".format(tstat=subjAve_roi_tstat_file)
        with open(onettest_args_fn, "a") as fo:
            fo.write(f"{subject} {coef_id} {tstat_id}\n")


def get_setAB(
    subject,
    subjAve_roi_coef_file,
    subjAve_roi_tstat_file,
    participants_df,
    setA,
    setB,
    program,
):
    sub_df = participants_df[participants_df["participant_id"] == subject]
    coef_id = "{coef}'[0]'".format(coef=subjAve_roi_coef_file)
    if (program == "3dttest++") or (program == "combat"):
        if sub_df["group"].values[0] == "control":
            setA.append(f"{subject} {coef_id}\n")
        elif sub_df["group"].values[0] == "case":
            setB.append(f"{subject} {coef_id}\n")
        else:
            pass
    elif program == "3dmema":
        tstat_id = "{tstat}'[0]'".format(tstat=subjAve_roi_tstat_file)
        if sub_df["group"].values[0] == "control":
            setA.append(f"{subject} {coef_id} {tstat_id}\n")
        elif sub_df["group"].values[0] == "case":
            setB.append(f"{subject} {coef_id} {tstat_id}\n")
        else:
            pass
    return setA, setB


def writearg_2sample(setA, setB, twottest_args_fn, program):
    setA = "".join(setA)
    setB = "".join(setB)
    with open(twottest_args_fn, "w") as fo:
        if (program == "3dttest++") or (program == "combat"):
            fo.write(f"-setA control\n{setA}-setB case\n{setB}")
        elif program == "3dmema":
            fo.write(f"-set control\n{setA}-set case\n{setB}")


def writecov_1sample(onettest_cov_fn, covariates_df, program):
    fix_labels = ["subj", "age"]
    gender_labels = [col for col in covariates_df if "gender" in col]

    if program == "combat":
        cov_labels = fix_labels + gender_labels
    else:
        site_labels = [col for col in covariates_df if "site" in col]
        cov_labels = fix_labels + gender_labels + site_labels

    with open(onettest_cov_fn, "w") as fo:
        fo.write("{}\n".format(" ".join(cov_labels)))


def write_combat_table(file_name):
    column_names = ["input", "output", "site", "group"]

    with open(file_name, "w") as fo:
        fo.write("{}\n".format(",".join(column_names)))


def write_table(table_fn_file):
    tab_labels = [
        "Subj",
        "group",
        "site",
        "age",
        "gender",
        "InputFile",
    ]
    with open(table_fn_file, "w") as fo:
        fo.write("{}\n".format("\t".join(tab_labels)))


def append2combat_table(
    subject,
    subjAve_roi_briks_file,
    subjAveHmz_roi_briks_file,
    participants_df,
    combat_table_fn,
):
    sub_df = participants_df[participants_df["participant_id"] == subject]

    sub_df = sub_df.fillna(0)
    group = sub_df["group"].values[0]
    site = sub_df["site"].values[0]

    variables = [
        subjAve_roi_briks_file,
        subjAveHmz_roi_briks_file,
        site,
        group,
    ]

    with open(combat_table_fn, "a") as fo:
        fo.write("{}\n".format(",".join(variables)))


def append2cov_1sample(subject, mean_fd, covariates_df, onettest_cov_fn, program):
    sub_df = covariates_df[covariates_df["participant_id"] == subject]
    age = sub_df["age"].values[0]
    fix_variables = [subject, age]
    gender_variables = [sub_df[col].values[0] for col in covariates_df if "gender" in col]

    if program == "combat":
        cov_variables = fix_variables + gender_variables
    else:
        site_variables = [sub_df[col].values[0] for col in covariates_df if "site" in col]
        cov_variables = fix_variables + gender_variables + site_variables

    cov_variables_str = [str(x) for x in cov_variables]

    with open(onettest_cov_fn, "a") as fo:
        fo.write("{}\n".format(" ".join(cov_variables_str)))


def append2table(subject, subjAve_roi_briks_file, participants_df, table_fn_file):
    sub_df = participants_df[participants_df["participant_id"] == subject]

    sub_df = sub_df.fillna(0)
    group = sub_df["group"].values[0]
    site = sub_df["site"].values[0]
    age = sub_df["age"].values[0]
    gender = sub_df["gender"].values[0].lower()
    InputFile = "{brik}[0]".format(brik=subjAve_roi_briks_file)
    cov_variables = [
        subject,
        group,
        site,
        age,
        gender,
        InputFile,
    ]

    cov_variables_str = [str(x) for x in cov_variables]
    with open(table_fn_file, "a") as fo:
        fo.write("{}\n".format("\t".join(cov_variables_str)))


def run_onesampttest(bucket_fn, mask_fn, covariates_file, args_file, program, n_jobs):
    if (program == "3dttest++") or (program == "combat"):
        cmd = f"3dttest++ -prefix {bucket_fn} \
                -mask {mask_fn} \
                -Covariates {covariates_file} \
                -Clustsim {n_jobs} \
                -ETAC {n_jobs} \
                -ETAC_opt NN=2:sid=2:hpow=0:pthr=0.05,0.01,0.005,0.001:name=etac \
                -@ < {args_file}"
    elif program == "3dmema":
        cmd = f"3dMEMA -prefix {bucket_fn} \
                -mask {mask_fn} \
                -covariates {covariates_file} \
                -verb 1 \
                -jobs {n_jobs} \
                -@ < {args_file}"
    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)


def run_twosampttest(bucket_fn, mask_fn, covariates_file, args_file, program, n_jobs):
    # with open(args_file) as file:
    #    arg_list = file.readlines()
    # arg_list_up = [x.replace("\n", "") for x in arg_list]
    # arg_list = " ".join(arg_list_up) # for 3dmema?
    if (program == "3dttest++") or (program == "combat"):
        cmd = f"3dttest++ -prefix {bucket_fn} \
                -AminusB \
                -mask {mask_fn} \
                -Covariates {covariates_file} \
                -Clustsim {n_jobs} \
                -ETAC {n_jobs} \
                -ETAC_opt NN=2:sid=2:hpow=0:pthr=0.05,0.01,0.005,0.001:name=etac \
                -@ < {args_file}"
    elif program == "3dmema":
        cmd = f"3dMEMA -prefix {bucket_fn} \
                -groups case control \
                -mask {mask_fn} \
                -covariates {covariates_file} \
                -verb 1 \
                -jobs {n_jobs} \
                -@ < {args_file}"

    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)


def run_onetwosampttest(bucket_fn, mask_fn, table_file, n_jobs):
    model = "'group+age+gender+(1|site)'"

    case_mean = "case_mean 'group : 1*case'"
    control_mean = "control_mean 'group : 1*control'"
    group_mean = "group_mean 'group : 0.5*case +0.5*control'"
    group_diff = "control-case 'group : 1*control -1*case'"
    cmd = f"3dLMEr -prefix {bucket_fn} \
        -mask {mask_fn} \
        -model {model} \
        -qVars 'age' \
        -qVarCenters '0' \
        -gltCode {case_mean} \
        -gltCode {control_mean} \
        -gltCode {group_mean} \
        -gltCode {group_diff} \
        -resid {bucket_fn}_res \
        -dbgArgs \
        -jobs {n_jobs} \
        -dataTable @{table_file}"

    print(f"\t\t{cmd}", flush=True)
    os.system(cmd)


def main(
    dset,
    deriv_dir,
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
    if program == "3dmema":
        roi_tstat_dict = {label: x * 3 + 2 for x, label in enumerate(roi_lst)}
    print(roi_coef_dict, flush=True)
    space = "MNI152NLin2009cAsym"
    n_jobs = int(n_jobs)
    # Load important tsv files
    participants_df = pd.read_csv(op.join(dset, "participants.tsv"), sep="\t")
    covariates_df = pd.read_csv(op.join(dset, "derivatives", "covariates.tsv"), sep="\t")

    # Define directories
    for session in sessions:
        if session is not None:
            rsfc_subjs_dir = op.join(rsfc_dir, "*", session, "func")
        else:
            # Include all the sessions if any
            rsfc_subjs_dir = op.join(rsfc_dir, "**", "func")
        session_label = ""

        rsfc_group_dir = op.join(rsfc_dir, f"group-{program}")
        os.makedirs(rsfc_group_dir, exist_ok=True)

        # Collect important files
        briks_files = sorted(
            glob(
                op.join(
                    rsfc_subjs_dir,
                    f"*task-rest*_space-{space}*_desc-norm_bucketREML+tlrc.HEAD",
                ),
                recursive=True,
            )
        )
        mask_files = sorted(
            glob(
                op.join(rsfc_subjs_dir, f"*task-rest*_space-{space}*_desc-brain_mask.nii.gz"),
                recursive=True,
            )
        )

        # Remove outliers using MRIQC metrics
        print(f"Total Briks: {len(briks_files)}, Masks: {len(mask_files)}", flush=True)
        clean_briks_files, clean_mask_files = remove_ouliers(mriqc_dir, briks_files, mask_files)
        print(
            f"MRIQC Briks: {len(clean_briks_files)}, Masks: {len(clean_mask_files)}",
            flush=True,
        )
        clean_briks_files, clean_mask_files = remove_missingdat(
            participants_df, clean_briks_files, clean_mask_files
        )
        print(
            f"Miss Briks: {len(clean_briks_files)}, Masks: {len(clean_mask_files)}",
            flush=True,
        )
        assert len(clean_briks_files) == len(clean_mask_files)
        # clean_briks_nm = [op.basename(x).split("_space-")[0] for x in clean_briks_files]
        # clean_mask_nm = [op.basename(x).split("_space-")[0] for x in clean_mask_files]
        # clean_briks_tpl = tuple(clean_briks_nm)
        # mask_not_brik = [x for x in clean_mask_nm if not x.startswith(clean_briks_tpl)]

        # Write group file
        clean_briks_fn = op.join(
            rsfc_dir,
            f"sub-group{session_label}_task-rest_space-{space}_briks.txt",
        )
        if not op.exists(clean_briks_fn):
            with open(clean_briks_fn, "w") as fo:
                for tmp_brik_fn in clean_briks_files:
                    fo.write(f"{tmp_brik_fn}\n")

        # Create group mask
        group_mask_fn = op.join(
            deriv_dir,
            f"sub-group{session_label}_task-rest_space-{space}_desc-brain_mask.nii.gz",
        )
        if not op.exists(group_mask_fn):
            if template_mask is None:
                template_mask_img = nib.load(clean_mask_files[0])
            else:
                template_mask_img = nib.load(template_mask)
            for clean_mask_file in clean_mask_files:
                clean_mask_img = nib.load(clean_mask_file)
                if clean_mask_img.shape != template_mask_img.shape:
                    clean_res_mask_img = image.resample_to_img(
                        clean_mask_img, template_mask_img, interpolation="nearest"
                    )
                    nib.save(clean_res_mask_img, clean_mask_file)

            group_mask = masking.intersect_masks(clean_mask_files, threshold=0.5)
            nib.save(group_mask, group_mask_fn)

        # Get template
        if template is None:
            # Resampling group to one subject
            for clean_briks_file in clean_briks_files:
                template = op.join(f"{clean_briks_file}'[{roi_coef_dict[roi]}]'")
                template_img = nib.load(template)
        else:
            template_img = nib.load(template)
        print(f"Using template {template} with size: {template_img.shape}", flush=True)

        roi_dir = op.join(rsfc_group_dir, roi)
        os.makedirs(roi_dir, exist_ok=True)

        if program != "3dlmer":
            setA = []
            setB = []
            # Conform onettest_args_fn and twottest_args_fn
            writearg_new_1sample = False
            writecov_new_1sample = False
            write_new_combat_table = False
            onettest_args_fn = op.join(
                roi_dir,
                f"sub-group{session_label}_task-rest_desc-1SampletTest{roi}_args.txt",
            )
            twottest_args_fn = op.join(
                roi_dir,
                f"sub-group{session_label}_task-rest_desc-2SampletTest{roi}_args.txt",
            )
            # Conform onettest_cov_fn and twottest_cov_fn
            onettest_cov_fn = op.join(
                roi_dir,
                f"sub-group{session_label}_task-rest_desc-1SampletTest{roi}_cov.txt",
            )
            if not op.exists(onettest_args_fn):
                writearg_new_1sample = True
                writearg_1sample(onettest_args_fn, program)

            if not op.exists(onettest_cov_fn):
                writecov_new_1sample = True
                writecov_1sample(onettest_cov_fn, covariates_df, program)

            if program == "combat":
                combat_table_fn = op.join(
                    roi_dir,
                    f"sub-group{session_label}_task-rest_desc-ComBat{roi}_table.txt",
                )
                if not op.exists(combat_table_fn):
                    write_new_combat_table = True
                    write_combat_table(combat_table_fn)

        else:
            write_new_table = False
            table_fn_file = op.join(
                roi_dir,
                f"sub-group{session_label}_task-rest_desc-2SampletTest{roi}_table.txt",
            )
            if not op.exists(table_fn_file):
                write_table(table_fn_file)
                write_new_table = True

        # Calculate subject and ROI level average connectivity
        subjects = [op.basename(x).split("_")[0] for x in clean_briks_files]
        subjects = sorted(list(set(subjects)))
        print(f"Group analysis sample size: {len(subjects)}")
        for subject in subjects:
            subj_briks_files = [x for x in clean_briks_files if subject in x]
            # This is an inelegant solution but it works for ALC108.
            # This take the session subj_briks_files according to the session from the
            # participants.tsv
            if session is None:
                select_first = False
                temp_ses_lst = [op.basename(x).split("_")[1] for x in subj_briks_files]
                for ses_i, temp_ses in enumerate(temp_ses_lst):
                    if temp_ses.split("-")[0] == "ses":
                        sub_df = participants_df[participants_df["participant_id"] == subject]
                        sub_df = sub_df.fillna(0)
                        if "session" in sub_df.columns:
                            select_ses = int(sub_df["session"].values[0])
                            if f"ses-{select_ses}" == temp_ses:
                                subj_briks_files = [subj_briks_files[ses_i]]
                                tmp_session = temp_ses
                                break
                        if temp_ses == temp_ses_lst[-1]:
                            select_first = True
                    else:
                        tmp_session = None
                # If the session from the participants.tsv was removed by the QC,
                # let's select the first session instead.
                if select_first:
                    subj_briks_files = [subj_briks_files[0]]
                    tmp_session = temp_ses_lst[0]
            else:
                tmp_session = session

            assert len(subj_briks_files) == 1

            if tmp_session is not None:
                rsfc_subj_dir = op.join(rsfc_dir, subject, tmp_session, "func")
                preproc_subj_dir = op.join(preproc_dir, subject, tmp_session, "func")
                clean_subj_dir = op.join(clean_dir, subject, tmp_session, "func")
            else:
                rsfc_subj_dir = op.join(rsfc_dir, subject, "func")
                preproc_subj_dir = op.join(preproc_dir, subject, "func")
                clean_subj_dir = op.join(clean_dir, subject, "func")

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
            subjAveHmz_roi_coef_file = op.join(
                rsfc_subj_dir,
                f"{prefix}_space-{space}_desc-ave{roi}hmz_coef",
            )
            if program == "3dmema":
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
                    clean_subj_dir,
                    subj_briks_files,
                    subjAve_roi_coef_file,
                    roi_coef_dict[roi],
                )
                if program == "3dmema":
                    # Average tstat across runs
                    subj_ave_roi(
                        clean_subj_dir,
                        subj_briks_files,
                        subjAve_roi_tstat_file,
                        roi_tstat_dict[roi],
                    )

            # Resampling
            subjAve_roi_briks = nib.load(f"{subjAve_roi_coef_file}+tlrc.BRIK")
            if subjAve_roi_briks.shape != template_img.shape:
                if not op.exists(f"{subjAveRes_roi_coef_file}+tlrc.BRIK"):
                    print(f"Resampling: {subjAve_roi_coef_file}")
                    conn_resample(
                        f"{subjAve_roi_coef_file}+tlrc",
                        subjAveRes_roi_coef_file,
                        template,
                    )
                    if program == "3dmema":
                        conn_resample(
                            f"{subjAve_roi_tstat_file}+tlrc",
                            subjAveRes_roi_tstat_file,
                            template,
                        )
                subjAve_roi_coef_file = subjAveRes_roi_coef_file
                subjAve_roi_tstat_file = "None"
                if program == "3dmema":
                    subjAve_roi_tstat_file = subjAveRes_roi_tstat_file

            # Multi-site harmonization
            if program == "combat":
                if op.exists(combat_table_fn) and write_new_combat_table:
                    append2combat_table(
                        subject,
                        f"{subjAve_roi_coef_file}+tlrc.BRIK",
                        subjAveHmz_roi_coef_file,
                        participants_df,
                        combat_table_fn,
                    )

                subjAve_roi_coef_file = subjAveHmz_roi_coef_file

            # Get subject level mean FD
            mean_fd = subj_mean_fd(preproc_subj_dir, subj_briks_files, subj_mean_fd_file)

            if program != "3dlmer":
                # Append subject specific info for onettest_args_fn
                if op.exists(onettest_args_fn) and writearg_new_1sample:
                    append2arg_1sample(
                        subject,
                        f"{subjAve_roi_coef_file}+tlrc.BRIK",
                        f"{subjAve_roi_tstat_file}+tlrc.BRIK",
                        onettest_args_fn,
                        program,
                    )

                # Get setA and setB to write twottest_args_fn
                if not op.exists(twottest_args_fn):
                    setA, setB = get_setAB(
                        subject,
                        f"{subjAve_roi_coef_file}+tlrc.BRIK",
                        f"{subjAve_roi_tstat_file}+tlrc.BRIK",
                        participants_df,
                        setA,
                        setB,
                        program,
                    )

                # Append subject specific info for onettest_cov_fn
                if op.exists(onettest_cov_fn) and writecov_new_1sample:
                    append2cov_1sample(subject, mean_fd, covariates_df, onettest_cov_fn, program)

            else:
                if op.exists(table_fn_file) and (write_new_table):
                    append2table(
                        subject,
                        f"{subjAve_roi_coef_file}+tlrc.BRIK",
                        participants_df,
                        table_fn_file,
                    )

        if program != "3dlmer":
            # Write twottest_args_fn
            if not op.exists(twottest_args_fn):
                writearg_2sample(setA, setB, twottest_args_fn, program)

            # Statistical analysis
            # Whole-brain, one-sample t-tests
            onettest_briks_fn = op.join(
                roi_dir,
                f"sub-group{session_label}_task-rest_desc-1SampletTest{roi}_briks",
            )

            if program == "combat":
                conn_harmonize(combat_table_fn, group_mask_fn)

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

            # Whole-brain, two-sample t-tests
            twottest_briks_fn = op.join(
                roi_dir,
                f"sub-group{session_label}_task-rest_desc-2SampletTest{roi}_briks",
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

        else:
            # Whole-brain, one-sample and two-sample t-tests
            onetwottest_briks_fn = op.join(
                roi_dir,
                f"sub-group{session_label}_task-rest_desc-1S2StTest{roi}_briks",
            )
            os.chdir(op.dirname(onetwottest_briks_fn))
            if not op.exists(f"{onetwottest_briks_fn}+tlrc.BRIK"):
                run_onetwosampttest(
                    op.basename(onetwottest_briks_fn),
                    group_mask_fn,
                    table_fn_file,
                    n_jobs,
                )


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
