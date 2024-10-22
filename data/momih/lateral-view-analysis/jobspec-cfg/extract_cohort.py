import argparse
import random
import re
from os.path import join, exists

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

labels_mapping = {
    'calcified granuloma': ['calcified densities', 'granuloma'],
    'calcified adenopathy': ['calcified densities', 'adenopathy', 'hilar enlargement'],
    'calcified mediastinal adenopathy ': ['calcified densities'],
    'calcified pleural thickening': ['calcified densities', 'pleural thickening'],
    'calcified pleural plaques': ['calcified densities', 'pleural plaques'],
    'heart valve calcified': ['calcified densities'],
    'calcified fibroadenoma': ['calcified densities'],
    'multiple nodules': ['nodule'],
    'nipple shadow': ['pseudonodule'],
    'end on vessel': ['end on vessel', 'pseudonodule'],
    'interstitial pattern': ['infiltrates'],
    'ground glass pattern': ['infiltrates'],
    'reticular interstitial pattern ': ['infiltrates'],
    'reticulonodular interstitial pattern': ['infiltrates'],
    'miliary opacities': ['infiltrates'],
    'alveolar pattern': ['infiltrates'],
    'consolidation': ['consolidation', 'infiltrates'],
    'air bronchogram': ['consolidation', 'infiltrates'],
    'total atelectasis': ['atelectasis'],
    'lobar atelectasis': ['atelectasis'],
    'segmental atelectasis': ['atelectasis'],
    'laminar atelectasis': ['atelectasis'],
    'round atelectasis': ['atelectasis'],
    'atelectasis basal': ['atelectasis'],
    'minor fissure thickening': ['fissure thickening'],
    'major fissure thickening': ['fissure thickening'],
    'loculated fissural effusion': ['fissure thickening', 'pleural effusion'],
    'apical pleural thickening': ['pleural thickening'],
    'loculated pleural effusion': ['pleural effusion'],
    'hydropneumothorax': ['pleural effusion', 'pneumothorax'],
    'empyema': ['pleural effusion'],
    'hemothorax': ['pleural effusion'],
    'central vascular redistribution': ['vascular redistribution'],
    'adenopathy': ['hilar enlargement'],
    'vascular hilar enlargement': ['hilar enlargement'],
    'pulmonary artery enlargement': ['hilar enlargement'],
    'descendent aortic elongation': ['aortic elongation', 'mediastinal enlargement'],
    'ascendent aortic elongation': ['aortic elongation', 'mediastinal enlargement'],
    'aortic button enlargement': ['aortic elongation'],
    'supra aortic elongation': ['aortic elongation', 'mediastinal enlargement'],
    'superior mediastinal enlargement': ['mediastinal enlargement'],
    'goiter': ['mediastinal enlargement'],
    'aortic aneurysm': ['mediastinal enlargement'],
    'mediastinal mass': ['mediastinal enlargement', 'mass'],
    'hiatal hernia': ['mediastinal enlargement', 'hernia'],
    'breast mass': ['mass'],
    'pleural mass': ['mass'],
    'pulmonary mass': ['mass'],
    'soft tissue mass': ['mass'],
    'scoliosis': ['thoracic cage deformation'],
    'kyphosis': ['thoracic cage deformation'],
    'pectum excavatum': ['thoracic cage deformation'],
    'pectum carinatum': ['thoracic cage deformation'],
    'cervical rib': ['thoracic cage deformation'],
    'vertebral compression': ['vertebral degenerative changes'],
    'vertebral anterior compression': ['vertebral degenerative changes'],
    'blastic bone lesion': ['sclerotic bone lesion'],
    'clavicle fracture': ['fracture'],
    'humeral fracture': ['fracture'],
    'vertebral fracture': ['fracture'],
    'rib fracture': ['fracture'],
    'callus rib fracture': ['fracture'],
    'central venous catheter': ['catheter'],
    'central venous catheter via subclavian vein': ['catheter'],
    'central venous catheter via jugular vein': ['catheter'],
    'reservoir central venous catheter': ['catheter'],
    'central venous catheter via umbilical vein': ['catheter'],
    'dual chamber device': ['electrical device'],
    'single chamber device': ['electrical device'],
    'pacemaker': ['electrical device'],
    'dai': ['electrical device'],
    'artificial mitral heart valve': ['artificial heart valve'],
    'artificial aortic heart valve': ['artificial heart valve'],
    'metal': ['surgery'],
    'osteosynthesis material': ['surgery'],
    'sternotomy': ['surgery'],
    'suture material': ['surgery'],
    'bone cement': ['surgery'],
    'prosthesis': ['surgery'],
    'humeral prosthesis': ['surgery'],
    'mammary prosthesis': ['surgery'],
    'endoprosthesis': ['surgery'],
    'aortic endoprosthesis': ['surgery'],
    'surgery breast': ['surgery'],
    'mastectomy': ['surgery'],
    'surgery neck': ['surgery'],
    'surgery lung': ['surgery'],
    'surgery heart': ['surgery'],
    'surgery humeral': ['surgery'],
    'atypical pneumonia': ['pneumonia'],
    'tuberculosis sequelae': ['tuberculosis'],
    'post radiotherapy changes': ['pulmonary fibrosis'],
    'asbestosis signs': ['pulmonary fibrosis'],
    'pulmonary artery hypertension': ['pulmonary hypertension'],
    'pulmonary venous hypertension': ['pulmonary hypertension']
}

cxr_labels = ['pneumonia', 'pleural effusion', 'consolidation', 'normal', 'cardiomegaly',
              'infiltrates', 'emphysema', 'mass', 'hernia', 'atelectasis',
              'pneumothorax', 'pulmonary edema', 'pleural thickening', 'nodule', 'pulmonary fibrosis']


def correct_image_dir(input_csv, output_csv, datadir, use_imagedir=False):
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"{len(df)} images in dataset.")

    # Some images don't exist
    def correct_image_dir_if_required(x):
        img_path = join(datadir, str(int(x['ImageDir'])), x['ImageID'])
        if not exists(img_path):
            x['ImageDir'] = -1
            for i in range(54):
                img_path = join(datadir, str(i), x['ImageID'])
                if exists(img_path):
                    x['ImageDir'] = i
                    break
        return x

    tqdm.pandas(desc="Correcting images path if needed")
    df = df.progress_apply(correct_image_dir_if_required, axis=1)

    def check_image_exists(x):
        return int(x['ImageDir']) != -1

    tqdm.pandas(desc="Dropping images that don't exist")
    images_exist = df[['ImageID', 'ImageDir']].progress_apply(check_image_exists, axis=1)
    print(f"{sum(~images_exist)} images don't exist and were dropped.")
    df = df.loc[images_exist]

    df.to_csv(output_csv, index=False)


def get_cohort(input_csv, output_csv, datadir, broken_images_file=None,
               mode='joint', correct=False):
    tqdm.pandas()
    random.seed(9999)

    if correct:
        correct_image_dir(input_csv, output_csv, datadir)

    usecols = ['ImageID', 'ImageDir', 'StudyDate_DICOM', 'StudyID', 'PatientID',
               'Projection', 'Pediatric', 'Rows_DICOM', 'Columns_DICOM', 'Labels']
    df = pd.read_csv(input_csv, usecols=usecols, low_memory=False)

    # Some pngs can't be read, we should remove them
    if broken_images_file is not None:
        with open(broken_images_file, 'r') as f:
            broken_images = f.readlines()
        broken_images = [im[:-1] for im in broken_images]

        df = df.loc[~df.ImageID.isin(broken_images)]

    # Only keeping those images that are L or PA and removing pediatric patients
    if mode == 'joint':
        df = df.loc[(df.Projection.isin(['L', 'PA'])) & (df.Pediatric == 'No')]
    elif mode == 'pa':
        df = df.loc[(df.Projection.isin(['PA'])) & (df.Pediatric == 'No')]

    # Removing duplicates
    df = df.drop_duplicates(subset=df.columns[1:]).drop(columns=['Pediatric'])

    # Removing images that don't have labels
    df = df.dropna(subset=['Labels'])

    # Keeping only those IDS that meet our criteria
    if mode == 'joint':
        tqdm.pandas(desc="Keeping only studies that have both L and PA")
    elif mode == 'pa':
        tqdm.pandas(desc="Keeping only studies that have PA")
    projs = df.groupby('StudyID').Projection.progress_apply(lambda x: ",".join(x))

    if mode == 'joint':
        ids_to_keep = projs[projs.isin(['PA,L', 'L,PA'])].index
    elif mode == 'pa':
        ids_to_keep = projs[projs.isin(['PA'])].index

    df = df.loc[df.StudyID.isin(ids_to_keep)]

    # Removing images whose label is 'exclude' or 'suboptimal study'
    labels_to_remove = re.compile('exclude|suboptimal study|unchanged')

    def remove_bad_labels(x):
        match = set(labels_to_remove.findall(x))
        return False if match else True

    tqdm.pandas(desc="Removing images with bad labels")
    good_labels = df.Labels.apply(remove_bad_labels)
    df = df.loc[good_labels]

    # Keeping only the first study for a patient
    tqdm.pandas(desc="Keeping only the first study for each patient")
    projs = df.groupby('PatientID').StudyDate_DICOM.progress_apply(lambda x: x.min())
    ids_to_keep = df.apply(lambda x: x.StudyDate_DICOM == projs[x.PatientID], axis=1)
    df = df.loc[ids_to_keep]

    # Some patients had multiple studies done the same day, so we pick randomly among those
    tqdm.pandas(desc="Choosing which study to keep for patients with multiple studies the same day")
    projs = df.groupby('PatientID').StudyID.progress_apply(lambda x: random.choice(x.tolist()))
    ids_to_keep = df.apply(lambda x: x.StudyID == projs[x.PatientID], axis=1)
    df = df.loc[ids_to_keep]

    # Map labels to the labels we care about
    def convert_labels(x):
        new_labels = [labels_mapping[label.strip()] if label.strip() in labels_mapping else [label.strip()]
                      for label in eval(x)]
        new_labels = [item for sublist in new_labels for item in sublist]  # flatten
        new_labels = list(set(new_labels))  # remove duplicates

        # Remove bad labels
        if '' in new_labels:
            new_labels.remove('')
        if 'chronic changes' in new_labels:
            new_labels.remove('chronic changes')
        return new_labels

    tqdm.pandas(desc="Mapping labels to their parent")
    df['Clean_Labels'] = df.Labels.progress_apply(convert_labels)

    # Remove images with non-existent labels
    def remove_bad_labels(x):
        return x != []

    tqdm.pandas(desc="Removing images with unmappable labels")
    good_labels = df.Clean_Labels.progress_apply(remove_bad_labels)
    df = df.loc[good_labels]

    # Removing images that don't have clean labels (typically, images with only 'chronic changes')
    df = df.dropna(subset=['Clean_Labels'])

    df.to_csv(output_csv, index=False)

    print(f"{len(df)} images in cohort from {len(df.PatientID.unique())} patients.")
    check_study_patient = (len(df.StudyID.unique()) == len(df.PatientID.unique()))
    print(f"Check if there is only one study per patient: {check_study_patient}")

    if mode == 'joint':
        check_study_image = (len(df.StudyID.unique()) * 2 == len(df))
        print(f"Check if there is are exactly two images per patient: {check_study_image}")

    return df


def labels_distribution(cohort):
    tqdm.pandas()

    df = pd.read_csv(cohort, low_memory=False)

    labels_dict = {}
    for labels in df.Clean_Labels:
        for label in eval(labels):
            label = label.strip()
            if label not in labels_dict:
                labels_dict[label] = 0
            labels_dict[label] += 1

    cxr_dict = {label: labels_dict[label] for label in cxr_labels}

    labels_list = []
    counts_list = []

    print('---------------------------------------------------')

    for k, v in sorted(cxr_dict.items(), key=lambda x: x[1], reverse=True):
        print(k, v // 2)
        labels_list.append(k)
        counts_list.append(v // 2)

    print('---------------------------------------------------')

    for k, v in sorted(labels_dict.items(), key=lambda x: x[1], reverse=True):
        if v > 100 and k not in cxr_labels:
            print(k, v // 2)
            labels_list.append(k)
            counts_list.append(v // 2)

    print('---------------------------------------------------')

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 13))

    sns.set_color_codes("muted")
    clrs = [sns.xkcd_rgb["medium green"] if (x < len(cxr_labels)) else sns.xkcd_rgb["denim blue"]
            for x in range(len(labels_list))]
    g = sns.barplot(x=counts_list, y=labels_list, palette=clrs)

    for index, row in enumerate(labels_list):
        g.text(counts_list[index] * 1.05, index, counts_list[index], color='black', va="center", fontsize=9)

    cxr_patch = mpatches.Patch(color=sns.xkcd_rgb["medium green"], label='CXR labels')
    pc_patch = mpatches.Patch(color=sns.xkcd_rgb["denim blue"], label='New labels')
    ax.legend(handles=[cxr_patch, pc_patch], ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="", xlabel="Number of patients")
    g.set_xscale('log')
    ax.set_title('Labels distribution for patients with both PA and L (N = {})'.format(len(df) // 2))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('data/labels_distribution.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage')
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--broken-file', default=None)
    parser.add_argument('--mode', default="joint")
    parser.add_argument('--joint-csv', default=None)
    parser.add_argument('--correct', action='store_true')
    args = parser.parse_args()

    mode = args.mode

    df = get_cohort(args.input_csv, args.output_csv, args.datadir,
                    args.broken_file, args.mode, args.correct)
    # labels_distribution(cohort_file)

    if args.mode == 'pa' and args.joint_csv is not None:
        # Remove common patients
        joint_df = pd.read_csv(args.joint_csv)
        diff_pt_ids = set(df.PatientID).difference(joint_df.PatientID)
        df = df.loc[df.PatientID.isin(diff_pt_ids)]
        df.to_csv(args.output_csv, index=False)
