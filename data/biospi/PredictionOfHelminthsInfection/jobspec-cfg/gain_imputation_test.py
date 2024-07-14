import argparse
import numpy as np
import plotly.express as px

import gain_imputation
from utils.Utils import create_rec_dir
import pandas as pd
from sys import exit
np.random.seed(0) #for reproducability
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--reshape', type=str)
    parser.add_argument('--w', type=str)
    parser.add_argument('--add_t_col', type=str)
    parser.add_argument('--thresh_daytime', type=str)
    parser.add_argument('--thresh_nan_ratio', type=str)

    args = parser.parse_args()
    print(args)

    NTOP = 17
    NJOB = 6
    ANSCOMBE = False
    LOG_ANSCOMBE = True
    REMOVE_ZEROS = False
    EXPORT_CSV = True
    EXPORT_TRACES = True
    ENABLE_FINAL_IMP = False
    WINDOW_ON = args.w.lower() in ["yes", 'y', 't', 'true']
    RESHAPE = args.reshape.lower() in ["yes", 'y', 't', 'true']
    ADD_T_COL = args.reshape.lower() in ["yes", 'y', 't', 'true']
    OUT = args.output_dir
    I_RANGE = 300
    THRESH_DT = int(args.thresh_daytime)
    THRESH_NAN_R = int(args.thresh_nan_ratio)


    # DATA_DIR = 'F:/Data2/backfill_1min_xyz_delmas_fixed'
    # DATA_DIR = 'backfill_1min_xyz_delmas_fixed'
    DATA_DIR = args.data_dir

    # config = [(WINDOW_ON, True, False, False), (WINDOW_ON, False, False, False), (WINDOW_ON, True, True, False), (WINDOW_ON, False, True, False), (WINDOW_ON, True, False, True), (WINDOW_ON, False, False, True)]
    config = [(WINDOW_ON, REMOVE_ZEROS, ANSCOMBE, LOG_ANSCOMBE)]
    for WINDOW_ON, REMOVE_ZEROS, ANSCOMBE, LOG_ANSCOMBE in config:

        OUT += '\imputation_test_window_%s_anscombe_%s_top%d_remove_zeros_%s_loganscombe_%s_reshape_%s' % (WINDOW_ON, ANSCOMBE, NTOP, REMOVE_ZEROS, LOG_ANSCOMBE, str(RESHAPE))
        # OUT = 'F:/Data2/imp_reshaped_full/imputation_test_window_%s_anscombe_%s_top%d_remove_zeros_%s_loganscombe_%s_debug' % (WINDOW_ON, ANSCOMBE, NTOP, REMOVE_ZEROS, LOG_ANSCOMBE)

        raw_data, original_data_x, ids, timestamp, date_str, ss_data = gain_imputation.load_farm_data(DATA_DIR, NJOB, NTOP, enable_remove_zeros=REMOVE_ZEROS,
                                                                                                      enable_anscombe=ANSCOMBE, enable_log_anscombe=LOG_ANSCOMBE, window=WINDOW_ON)

        hist_array = raw_data.flatten()
        hist_array_nrm = hist_array[~np.isnan(hist_array)]
        hist_array_zrm = hist_array_nrm[hist_array_nrm > 1]

        df = pd.DataFrame(hist_array_zrm, columns=["value"])
        print(df)
        fig = px.histogram(df, x="value", nbins=np.unique(hist_array_zrm).size)
        filename = args.output_dir + "/" + "histogram_raw_input_zrm.html"
        create_rec_dir(filename)
        fig.write_html(filename)

        df = pd.DataFrame(hist_array_nrm, columns=["value"])
        print(df)
        fig = px.histogram(df, x="value", nbins=np.unique(hist_array_nrm).size)
        filename = args.output_dir + "/" + "histogram_raw_input.html"
        fig.write_html(filename)

        missing_range = [0.0]

        for i, miss_rate in enumerate(missing_range):
            rmse_list = []
            rmse_list_li = []
            # for i, i_r in enumerate(iteration_range):
            print("progress %d/%d..." % (i, len(missing_range)))
            parser = argparse.ArgumentParser()
            parser.add_argument('--data_dir', type=str, default=DATA_DIR)
            parser.add_argument('--output_dir', type=str, default=OUT)
            parser.add_argument(
                '--batch_size',
                help='the number of samples in mini-batch',
                default=128,
                type=int)
            parser.add_argument(
                '--hint_rate',
                help='hint probability',
                default=0.9,
                type=float)
            parser.add_argument(
                '--alpha',
                help='hyperparameter',
                default=100,
                type=float)
            parser.add_argument(
                '--iterations',
                help='number of training interations',
                default=I_RANGE,
                type=int)
            parser.add_argument(
                '--miss_rate',
                help='missing data probability',
                default=miss_rate,
                type=float)
            parser.add_argument('--n_job', type=int, default=NJOB, help='Number of thread to use.')
            parser.add_argument('--n_top_traces', type=int, default=NTOP,
                                help='select n traces with highest entropy (<= 0 number to select all traces)')
            parser.add_argument('--enable_anscombe', type=bool, default=ANSCOMBE)
            parser.add_argument('--export_csv', type=bool, default=EXPORT_CSV)
            parser.add_argument('--export_traces', type=bool, default=EXPORT_TRACES)
            parser.add_argument('--reshape', type=str, default=RESHAPE)
            parser.add_argument('--w', type=str, default=WINDOW_ON)
            parser.add_argument('--add_t_col', type=str, default=ADD_T_COL)
            parser.add_argument('--thresh_daytime', type=str, default=THRESH_DT)
            parser.add_argument('--thresh_nan_ratio', type=str, default=THRESH_NAN_R)

            args = parser.parse_args()
            print(args)
            gain_imputation.main(args, raw_data, original_data_x, ids, timestamp, date_str, ss_data)


