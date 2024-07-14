from utils import audio_visual, stats_agg, stats_single, load_dataset, load_datasets, load_datasets_b, stats_agg_mult, get_train_test, siamese_models, siamese_models_conv_dropout
import numpy as np
from process_audio import rcs_single, audio_single, audio_single_paper, AudioDataset, AudioPair, into_full_phoneme
from process_audio import balance_labels, RCSDataset, RCSPair, balance_labels_agg, preprocess_agg_it, make_RCSPair
from process_audio import create_pairs, balance_labels_agg_mult, balance_labels_mult
from test_siamese import test_loop
from tunes import margin_threshold_siamese, margin_threshold_siamese_agg, margin_threshold_multiple, margin_threshold_multiple_models, margin_threshold_multiple_models_cv
from tunes import margin_threshold_multiple_models_cv_margins, siamese_model_train, margin_threshold_multiple_single_margin
from draw import plot_reg
from siamese import SiameseNetwork

import pickle
import time

import torch
from torch.utils.data import DataLoader
from torch import optim

from process_audio import preprocess_single

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Variables
SR = np.float32(16000)
N = np.float32(16)
# RCS = np.zeros(np.int32(N), dtype=np.float32)
RCS = np.array([-0.7, 0.0, 0.4, -0.5, 0.5, -0.3, 0.4, 0.0, 0.4, 0.1, 0.3, 0.4, 0.1, -0.1, 0.2, 0.0], dtype=np.float32)
OFFSET = 0.01
EPOCHS = 1000
THRESHOLD_E = 0.1
L_TUBE = 17.5
V_SOUND = 35000
TAU = L_TUBE / (V_SOUND * N)    # tau = T / 2 = 3.125e-5
                                # = L_TUBE / (V_SOUND * N)
                                # L_TUBE / V_SOUND = 5e-4
                                # N = 16
# print(f"TAU: {TAU}")
# print(f"TAU^-1: {1 / TAU}")
THRESHOLD_VC = 0.001
BATCH_SIZE = 16
PRED_TRESHOLDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
MARGINS = [1, 2, 5, 10, 20, 30]
MARGINS_LARGER = [40, 50, 60, 70, 80, 90, 100]
MARGIN = 10
NUM_AGGS = [2, 5, 10]
MODELS = siamese_models()
MODEL_NAMES = ["SiameseNetwork", "Siamese_dropout", "Siamese_dropout_hidden", "Siamese_st16", "Siamese_st8", "Siamese_fc", "Siamese_Conv", "Siamese_Conv_fc"]
MODELS_CONV_DROPOUT = siamese_models_conv_dropout()
MODEL_NAMES_CONV_DROPOUT = ["Siamese_conv1_dropout", "Siamese_conv2_dropout"]

SINGLE_TRAIN = 'data/processed/train_w_rcs.pkl'
SINGLE_TEST = 'data/processed/test_w_rcs.pkl'
AGG_TRAIN = 'data/processed/train_agg.pkl'
AGG_TEST = 'data/processed/test_agg.pkl'
SINGLE_TRAIN_B = 'data/processed/train_b'
SINGLE_TEST_B = 'data/processed/test_b'


# CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Phoneme categories
stops = {"b", "d", "g", "p", "t", "k", "dx", "q"}
affricates = {"jh", "ch"}
fricatives = {"s", "sh", "z", "zh", "f", "th", "v", "dh"}
nasals = {"m", "n", "ng", "em", "en", "eng", "nx"}
semivowels_glides = {"l", "r", "w", "y", "hh", "hv", "el"}
vowels = {"iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"}
others = {"pau", "epi", "h#", "1", "2"}

# audio_visual("SA1.WAV.wav", "SA1.PHN", SR, vowels)

# results = audio_single(RCS, EPOCHS, SR, THRESHOLD_VC, N, "SA1.WAV.wav", "SA1.PHN", vowels, OFFSET)

def main():
    # Data Preprocessing
    # Extract RCS from audios and create datasets
    # preprocess_single(RCS, EPOCHS, SR, THRESHOLD_VC, N, vowels, OFFSET, SINGLE_TRAIN, SINGLE_TEST)

    # Create aggregated dataset
    # dataset_train_single, dataset_test_single = load_dataset(SINGLE_TRAIN, SINGLE_TEST)
    # preprocess_agg_it(dataset_train_single, dataset_test_single, NUM_AGGS)

    # train_tests_paired = balance_labels_agg_mult(train_tests_paired)
    # agg_test = balance_labels_agg(agg_test)
    # print(f"Original Training set length: {len(train_paired)}")
    # print(f"Original test set length: {len(test_paired)}")
    # print(f"Balanced Training Set length: {len(train_paired_b)}")
    # print(f"Balanced Test Set length: {len(test_paired_b)}")
    # print("agg_train: ", len(agg_train[0]))
    # print("agg_test: ", agg_test[0])
    # print(f"agg_train len: {len(agg_train)}")
    # print(f"agg_test len: {len(agg_test)}")

    # Load datasets
    # Create Paired Dataset
    # train, test = load_dataset(SINGLE_TRAIN, SINGLE_TEST)
    # train_paired, test_paired = create_pairs(load_dataset(SINGLE_TRAIN, SINGLE_TEST))

    # Aggregated datasets (for agg 2, 5, 10)
    # train_tests_paired = make_RCSPair(load_datasets(NUM_AGGS))
    # train_b, test_b = load_dataset(SINGLE_TRAIN_B, SINGLE_TEST_B)
    # print(f"train_b lenth: {len(train_b)}")
    # print(f"test_b length: {len(test_b)}")

    # Aggregated, Balanced Datasets for different NUM_AGG
    train_tests_paired = load_datasets_b(NUM_AGGS)
    # Pick a agg_num and run ablation test
    # num_agg, train_b, test_b = get_train_test(train_tests_paired, 2)

    # Truncate Datasets
    # train_paired_b = balance_labels(train_paired)
    # agg_train = balance_labels_agg_mult(train_tests_paired)
    # test_paired_b = balance_labels(test_paired)
    # balance_labels_mult(train_paired, test_paired, SINGLE_TRAIN_B, SINGLE_TEST_B)
    # train_tests_paired = balance_labels_agg_mult(train_tests_paired)

    # Anaylitics
    # stats_single(train, test)
    # stats_agg_mult(train_tests_paired)
    # plot_reg()

    # print(agg_train[0])

    siamese = SiameseNetwork()
    # dataloader_train = DataLoader(train_b, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # dataloader_test = DataLoader(test_b, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # For paper, single audio outputs
    # audio_single_paper(RCS, EPOCHS, SR, THRESHOLD_VC, N, "data/Train/DR4/FALR0/SA1.WAV.wav", "data/Train/DR4/FALR0/SA1.PHN", "data/Train/DR4/FALR0/SA1.TXT",vowels, OFFSET)
    
    # Hyperparameter Tuning
    # margin_threshold_siamese(MARGINS, siamese, dataloader_train, dataloader_test, 
    #                          optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, 
    #                          DEVICE)
    # margin_threshold_multiple(MARGINS, siamese, optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, 
    #                          DEVICE, train_tests_paired, BATCH_SIZE)
    # margin_threshold_multiple_models(MARGINS, MODELS, MODEL_NAMES, EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, 
    #                           DEVICE, BATCH_SIZE, 2, train_b, test_b)
    
    # cross validation of various model architectures
    # margin_threshold_multiple_models_cv(1, MODELS, MODEL_NAMES, EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, 
    #                           DEVICE, BATCH_SIZE, 2, train_b)
    
    # CV for margins
    # margin_threshold_multiple_models_cv_margins(MARGINS + MARGINS_LARGER, siamese, "SiameseNetwork", EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, DEVICE, BATCH_SIZE, 2, train_b)

    # Network = SiameseNetwork, margin = 20
    # siamese_model_train(MARGIN, siamese, dataloader_train, dataloader_test, optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, 
    #                          THRESHOLD_VC, N, vowels, OFFSET, DEVICE, 1, "SiameseNetwork")
    
    # final training with single margin and multiple agg_nums
    margin_threshold_multiple_single_margin(MARGIN, siamese, optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, DEVICE, 
                                            train_tests_paired, BATCH_SIZE, "SiameseNetwork")

if __name__ == "__main__":
    main()
