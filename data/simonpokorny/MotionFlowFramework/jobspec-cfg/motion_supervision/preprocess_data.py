from motion_supervision import constants as C

from motion_supervision.generate_priors import *
from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence
from my_datasets.kitti.semantic_kitti import SemanticKitti_Sequence
from my_datasets.waymo.waymo import Waymo_Sequence
from my_datasets.paths import *
from motion_supervision.ego_utils import generate_ego_priors_for_sequence

import sys

if __name__ == '__main__':
    # loop over used datasets, when ready
    seq = int(sys.argv[1])

    datasets = ['waymo']

    init_preprocessing = False

    for dataset in datasets:

        if dataset == 'semantic_kitti':
            try:
                sequence = SemanticKitti_Sequence(sequence_nbr=seq)
            except:
                print(f"Sequence {seq} does not exist in Semantic Kitti, Mapping is not yet fixed")
                continue

        elif dataset == 'waymo':
            try:
                sequence = Waymo_Sequence(sequence_nbr=seq)
            except:
                print(f"Sequence {seq} does not exist in Waymo")
                continue

        elif dataset == 'argoverse2':
            try:
                sequence = Argoverse2_Sequence(sequence_nbr=seq)
            except:
                print(f"Sequence {seq} does not exist in Argoverse2")
                continue

        else:
            raise ValueError("Dataset not supported")

        # if dataset == 'waymo':

        # if dataset == 'semantic_kitti':
        # print(f'Generating Ego priors for {dataset} sequence: {seq}')
        # generate_ego_priors_for_sequence(sequence, C.cfg)

        # print(f"Generating static priors values for {dataset} sequence: {seq}")
        # generate_static_points_from_sequence(sequence, C.cfg)

        print(f"Generating freespace for {dataset} sequence: {seq}")
        generate_freespace_from_sequence(sequence, C.cfg)

        print(f"Generating visibility priors for {dataset} sequence: {seq}")
        generate_visibility_prior(sequence, C.cfg)

        print(f"Generating corrected priors if they were static at some point for {dataset} sequence: {seq}")
        correct_the_dynamic_priors(sequence, C.cfg)

        print(f"Generating full body dynamic objects from corrected priors")    # merge?
        project_dynamic_label_to_cell(sequence, C.cfg)

        # print(f"Saving MOS format")
        # store_final_priors_in_mos_format(sequence)


