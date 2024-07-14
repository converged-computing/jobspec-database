# Copyright 2018
# 
# Yaojie Liu, Amin Jourabloo, Xiaoming Liu, Michigan State University
# 
# All Rights Reserved.
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import Config
from model.dataset import Dataset
from model.model import Model


def main(argv=None):
    # Configurations
    config = Config()
    config.DATA_DIR = ['./data/SiW_M_Makeup_Ob_Binary_Files',
                       './data/SiW_M_Mask_Silicone_Binary_Files',
                       './data/SiW_M_Makeup_Co_Binary_Files',
                       './data/SiW_M_Mask_Paper_Binary_Files',
                       './data/SiW_M_Makeup_Im_Binary_Files',
                       './data/SiW_M_Mask_Mann_Binary_Files',
                       './data/SiW_M_Replay_Binary_Files',
                       './data/SiW_M_Partial_Cut_Binary_Files',
                       './data/SiW_M_Mask_Half_Binary_Files',
                       './data/SiW_M_Partial_Funnyeye_Binary_Files',
                       './data/SiW_M_Partial_Paperglass_Binary_Files',
                       './data/SiW_M_Mask_Trans_Binary_Files',
                       './data/SiW_M_Paper_Binary_Files',
                       './data/SiW_M_Live_Binary_Files',
                       './data/SiW_M_Live_Test_Binary_Files']
    config.DATA_DIR_LIVE = ['./data/SiW_M_Live_Binary_Files']
    config.DATA_DIR_TEST = ['./data/SiW_M_Live_Test_Binary_Files']
    config.LOG_DIR = './logs/model'
    config.MODE = 'training'
    # config.MODE = 'testing'
    config.STEPS_PER_EPOCH_VAL = 180
    config.display()

    # Get images and labels.
    # dataset_train = Dataset(config, 'train')
    # Build a Graph
    model = Model(config)

    # Train the model
    model.compile()
    model.train()
    # model.test()


main()
