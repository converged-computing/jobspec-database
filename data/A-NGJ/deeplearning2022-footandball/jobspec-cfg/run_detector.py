# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

# Detector developed further on Soccer Net tracking dataset
# Emilia Konopko, Barbara Pierkarska, Aleksander Nagaj, Piotr Saffarini
# As a part of 02456 Deep Learning at DTU

import logging

import os
from pathlib import Path
from pprint import pprint
import sys
import typing as t
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import tqdm

from image import metric, image
from misc import utils
from network import footandball
from data import soccer_net, augmentation

TEST_DIR = os.path.expandvars("${REPO}/runs/test")

logging.basicConfig(format="%(levelname)s: %(message)s")

# pylint: disable=too-many-locals
def run_detector(model, args) -> t.Optional[np.array]:
    """
    Runs FootAndBall detector.

    Returns
    -------
    Optional[np.array] :
        Intersection of union metric, if specified in args
    """

    soccer_net_ = None
    start_frame = 0

    if args.metric_path:
        start_frame = int(Path(args.path).name.split(".")[0])
        metric_path = Path(args.metric_path)
        soccer_net_ = soccer_net.SoccerNet(metric_path.parents[0])
        soccer_net_.collect([metric_path.name])
        if len(soccer_net_.gt) == 0:
            raise EnvironmentError(
                f"missing ground truth labels in {metric_path.name} directory"
            )

    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    if args.device == "cpu":
        print("Loading CPU weights...")
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print("Loading GPU weights...")
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    sequence = cv2.VideoCapture(args.path)
    fps = sequence.get(cv2.CAP_PROP_FPS)
    if args.metric_path:
        sequence = cv2.VideoCapture(args.path, cv2.CAP_IMAGES)
        fps = 30

    (frame_width, frame_height) = (
        int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print("width: ", frame_width, "height: ", frame_height)
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))

    out_sequence = cv2.VideoWriter(
        args.out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    print(f"Processing video: {args.path}")
    pbar = tqdm.tqdm(total=n_frames)

    # Intersection over Union
    iou = []
    # Ratio of number of detected players to number of ground truth labels
    detected = []
    while sequence.isOpened():

        ret, frame = sequence.read()
        if not ret:
            # End of video
            break
        # Convert color space from BGR to RGB, convert to tensor and normalize
        img_tensor = augmentation.numpy2tensor(frame)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
            detections = model(img_tensor)[0]

            if args.metric_path:
                gt, gt_detected = metric.getGT(soccer_net_.gt[start_frame - 1])

                det, det_detected = metric.getGT(detections["boxes"])

                if gt_detected == 0:
                    detected.append(1)
                else:
                    detected.append(det_detected / gt_detected)
                iou.append(metric.IoU(det, gt))

        # Display overlap of detection and gt for debug purposes
        if args.debug:
            frameGT = image.draw_bboxes(
                frame, soccer_net_.gt[start_frame - 1], image.Color.GREEN
            )
            frameGT = image.draw_bboxes(frameGT, detections["boxes"], image.Color.RED)
            cv2.imshow("img", frameGT)
            if cv2.waitKey(0) == 27:
                args.debug = False
            cv2.destroyAllWindows()

        frame = image.draw_bboxes_on_detections(frame, detections)
        out_sequence.write(frame)
        pbar.update(1)
        start_frame += 1

    pbar.close()
    sequence.release()
    out_sequence.release()

    return iou, detected


def main():
    if not "DATA_PATH" in os.environ:
        logging.error("missing DATA_PATH environmental variable")
        return 1

    if not "REPO" in os.environ:
        logging.error("missing REPO environmental variable")
        return 1

    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser(
        description="Run FootAndBall detector on input video"
    )
    parser.add_argument(
        "--path", help="path to video/images used for detction", type=str, required=True
    )
    parser.add_argument("--model", help="model name", type=str, default="fb1")
    parser.add_argument(
        "--weights", help="path to model weights", type=str, required=True
    )
    parser.add_argument(
        "--ball-threshold",
        help="ball confidence detection threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--player-threshold",
        help="player confidence detection threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "-o",
        "--out-video",
        help="path to video with detection results",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "-d", "--device", help="device (CPU or CUDA)", type=str, default="cuda:0"
    )
    parser.add_argument(
        "--run-dir",
        help="[Optional] Directory for saving test data; default: YYMMDD_HHMM",
        required=False,
        default=utils.get_current_time(),
    )
    parser.add_argument(
        "-m",
        "--metric-path",
        help="Ground truth dir. If used metrics are calculated among detection",
    )
    parser.add_argument(
        "--debug",
        help="Debug mode. Displays overlap of detection and ground truth during detection",
        action="store_true",
    )

    args = parser.parse_args()

    print()
    print("=" * 20 + " Running FootAndBall detection " + "=" * 20)
    pprint(args.__dict__)
    print("=" * 71)
    print()

    try:
        assert os.path.exists(
            args.weights,
        ), f"Cannot find FootAndBall model weights: {args.weights}"
        assert os.path.exists(args.path), f"Cannot open video: {args.path}"
    except AssertionError as err:
        logging.error(err)
        return 1

    model = footandball.model_factory(
        args.model,
        "detect",
        ball_threshold=args.ball_threshold,
        player_threshold=args.player_threshold,
    )

    # general run history directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    # run specific history directory
    run_dir = f"{TEST_DIR}/{args.run_dir}"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    try:
        with open(
            os.path.join(run_dir, "run_parameters.json"), "w", encoding="utf-8"
        ) as wfile:
            json.dump(args.__dict__, wfile, indent=2)
    except EnvironmentError as err:
        logging.error(err)
        return 1

    args.out_video = os.path.join(run_dir, args.out_video)
    # RUN DETECTOR
    try:
        metric, detected = run_detector(model, args)
    # pylint: disable=broad-except
    except Exception as err:
        logging.error(err)
        return 1

    if len(metric) > 0:
        try:
            with open(
                os.path.join(run_dir, "iou.json"), "w", encoding="utf-8"
            ) as outfile:
                json.dump(
                    {
                        "raw": metric,
                        "avg": np.mean(metric),
                        "std": np.std(metric),
                        "detected_ratio": np.mean(detected),
                    },
                    outfile,
                    indent=2,
                )
        except EnvironmentError as err:
            logging.error(err)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
