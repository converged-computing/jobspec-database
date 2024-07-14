import argparse
import datetime
import json
import os
import typing

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm

from dataloader.dataloader import prepare_dataloader as prepare_dataloader_t06
from models import prepare_model, prepare_model_eval
from utils.rescale_GHI import rescale_GHI


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    data_loader = prepare_dataloader_t06(
        dataframe,
        target_datetimes,
        stations,
        target_time_offsets,
        config
    )

    return data_loader


def prepare_model(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    model = prepare_model_eval(
        stations,
        target_time_offsets,
        config
    )

    return model


def generate_predictions(data_loader: tf.data.Dataset, model: tf.keras.Model, pred_count: int) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""
    predictions = []
    with tqdm.tqdm("generating predictions", total=pred_count) as pbar:
        for iter_idx, minibatch in enumerate(data_loader):
            assert isinstance(minibatch, tuple) and len(minibatch) >= 2, \
                "the data loader should load each minibatch as a tuple with model input(s) and target tensors"
            # remember: the minibatch should contain the input tensor(s) for the model as well as the GT (target)
            # values, but since we are not training (and the GT is unavailable), we discard the last element
            # see https://github.com/mila-iqia/ift6759/blob/master/projects/project1/datasources.md#pipeline-formatting
            if len(minibatch) == 2:  # there is only one input + groundtruth, give the model the input directly
                pred = model(minibatch[0])
            else:  # the model expects multiple inputs, give them all at once using the tuple
                pred = model(minibatch[:-1])
            if isinstance(pred, tf.Tensor):
                pred = pred.numpy()
            pred = rescale_GHI(pred)
            assert pred.ndim == 2, "prediction tensor shape should be BATCH x SEQ_LENGTH"
            predictions.append(pred)
            pbar.update(len(pred))
    return np.concatenate(predictions, axis=0)


def generate_all_predictions(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any],
) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""
    # we will create one data loader per station to make sure we avoid mixups in predictions
    predictions = []
    for station_idx, station_name in enumerate(target_stations):
        # usually, we would create a single data loader for all stations, but we just want to avoid trouble...
        stations = {station_name: target_stations[station_name]}
        print(f"preparing data loader & model for station '{station_name}' ({station_idx + 1}/{len(target_stations)})")
        data_loader = prepare_dataloader(dataframe, target_datetimes, stations, target_time_offsets, user_config)
        model = prepare_model(stations, target_time_offsets, user_config)
        station_preds = generate_predictions(data_loader, model, pred_count=len(target_datetimes))
        assert len(station_preds) == len(target_datetimes), "number of predictions mismatch with requested datetimes"
        predictions.append(station_preds)
    return np.concatenate(predictions, axis=0)


def parse_gt_ghi_values(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station GHI values from the provided dataframe for the evaluation of predictions."""
    gt = []
    for station_idx, station_name in enumerate(target_stations):
        station_ghis = dataframe[station_name + "_GHI"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_ghis.index:
                    seq_vals.append(station_ghis.iloc[station_ghis.index.get_loc(index)])
                else:
                    seq_vals.append(float("nan"))
            gt.append(seq_vals)
    return np.concatenate(gt, axis=0)


def parse_nighttime_flags(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        station_flags = dataframe[station_name + "_DAYTIME"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_flags.index:
                    seq_vals.append(station_flags.iloc[station_flags.index.get_loc(index)] > 0)
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def main(
        preds_output_path: typing.AnyStr,
        admin_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
        stats_output_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    """Extracts predictions from a user model/data loader combo and saves them to a CSV file."""

    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    if "start_bound" in admin_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
    if "end_bound" in admin_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]

    target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]

    if "bypass_predictions_path" in admin_config and admin_config["bypass_predictions_path"]:
        # re-open cached output if possible (for 2nd pass eval)
        assert os.path.isfile(preds_output_path), f"invalid preds file path: {preds_output_path}"
        with open(preds_output_path, "r") as fd:
            predictions = fd.readlines()
        assert len(predictions) == len(target_datetimes) * len(target_stations), \
            "predicted ghi sequence count mistmatch wrt target datetimes x station count"
        assert len(predictions) % len(target_stations) == 0
        predictions = np.asarray([float(ghi) for p in predictions for ghi in p.split(",")])
    else:
        predictions = generate_all_predictions(target_stations, target_datetimes,
                                               target_time_offsets, dataframe, user_config)
        with open(preds_output_path, "w") as fd:
            for pred in predictions:
                fd.write(",".join([f"{v:0.03f}" for v in pred.tolist()]) + "\n")

    if any([s + "_GHI" not in dataframe for s in target_stations]):
        print("station GHI measures missing from dataframe, skipping stats output")
        return

    assert not np.isnan(predictions).any(), "user predictions should NOT contain NaN values"
    predictions = predictions.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    gt = parse_gt_ghi_values(target_stations, target_datetimes, target_time_offsets, dataframe)
    gt = gt.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    day = parse_nighttime_flags(target_stations, target_datetimes, target_time_offsets, dataframe)
    day = day.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))

    squared_errors = np.square(predictions - gt)
    stations_rmse = np.sqrt(np.nanmean(squared_errors, axis=(1, 2)))
    for station_idx, (station_name, station_rmse) in enumerate(zip(target_stations, stations_rmse)):
        print(f"station '{station_name}' RMSE = {station_rmse:.02f}")
    horizons_rmse = np.sqrt(np.nanmean(squared_errors, axis=(0, 1)))
    for horizon_idx, (horizon_offset, horizon_rmse) in enumerate(zip(target_time_offsets, horizons_rmse)):
        print(f"horizon +{horizon_offset} RMSE = {horizon_rmse:.02f}")
    overall_rmse = np.sqrt(np.nanmean(squared_errors))
    print(f"overall RMSE = {overall_rmse:.02f}")

    if stats_output_path is not None:
        # we remove nans to avoid issues in the stats comparison script, and focus on daytime predictions
        squared_errors = squared_errors[~np.isnan(gt) & day]
        with open(stats_output_path, "w") as fd:
            for err in squared_errors.reshape(-1):
                fd.write(f"{err:0.03f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_out_path", type=str,
                        help="path where the raw model predictions should be saved (for visualization purposes)")
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    parser.add_argument("-s", "--stats_output_path", type=str, default=None,
                        help="path where the prediction stats should be saved (for benchmarking)")
    args = parser.parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
        stats_output_path=args.stats_output_path,
    )
