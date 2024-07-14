#! /usr/bin/env python3

import os
import sys
import argparse
import json
from collections import namedtuple
from time import time

import cv2
import numpy as np
import tensorflow
import csv
from PIL import Image, ImageOps

import utils.utils as utils
from detector import FaceNetDetector, MTCNNDetector, RetinaFaceDetector
from sort import Sort
from scene import SceneChangeDetector

from keras_facenet import FaceNet
from keras_facenet.utils import cropBox

facenet_models = [ '20180402-114759', '20180408-102900',
                   '20170511-185253', '20170512-110547' ]

FACE_IMAGE_SIZE = 160  # save face crops in this image resolution! (required!)
SAVE_FACE_PADDING = 0.10  # before saving to disk, add this padding to show more face

Options = namedtuple(
    "Options",
    ["out_path", "n_shards", "shard_i", "save_every", "min_trajectory",
     "display_width", "display_height", "max_trajectory_age", "save_images"],
)

debug = False

def bbox_float_to_int(bbox_float, max_w, max_h, padding=0.0):
    """Convert float bounding box to integers.
    """
    bbox_float = np.array(bbox_float, dtype=np.float32)

    # Optionally pad to show more of the face
    dim = np.minimum(bbox_float[2] - bbox_float[0], bbox_float[3] - bbox_float[1])
    pad_px = padding * dim
    bbox_float += np.array([-pad_px, -pad_px, pad_px, pad_px])

    # Make sure box doesn't extend beyond image borders
    bbox_float = np.maximum(bbox_float, [0, 0, 0, 0])
    bbox_float = np.minimum(bbox_float, [max_w, max_h, max_w, max_h])
    bbox_float = np.round(bbox_float)

    return [int(c) for c in bbox_float]

def save_trajectories(file, trackers, max_w, max_h):
    """Save trajectories from all given trackers, to file.
    """
    # Extract trajectories from trackers and write to jsonl
    for trk in trackers:
        trajectory = []
        detected = []
        for bbox_float, d in trk.history:
            bbox_int = bbox_float_to_int(bbox_float, max_w, max_h)
            trajectory.append(bbox_int)
            detected.append(d)

        # Note: `index` and `movie_id` keys are added in `merge_shards.py`
        out_obj = {
            "start": trk.first_frame,
            "len": len(trajectory),
            "bbs": trajectory,
            "detected": detected,
            "w": max_w,
            "h": max_h,
            "object_type": "face",
        }
        json.dump(out_obj, file, indent=None, separators=(",", ":"))
        file.write("\n")

    return len(trackers)

# to be obsoleted:
def process_frame_old(frame_data, d_width, d_height, features_file, images_dir,
                      min_trajectory_len, save_image):
    """Save faces + features from a frame, and creating face embeddings.
    """
    if debug:
        for face in frame_data["faces"]:
            print(multi_tracker.has_valid_tracker_safe(face["detection_id"]), face)
    # Filter to faces with a valid trajectory (len > MIN)
    valid_faces = [
        face for face in frame_data["faces"]
        if multi_tracker.has_valid_tracker(face["detection_id"])
    ]

    img = Image.fromarray(frame_data["img_np"])
    for face in valid_faces:
        # Retrieve the posterior bbox filtered by the Kalman filter
        filtered_box = multi_tracker.get_detection_bbox(face["detection_id"])

        # Crop onto face only (tight crop for embedding)
        tight_box = bbox_float_to_int(filtered_box, d_width, d_height)
        cropped = img.crop(tuple(tight_box))
        resized = cropped.resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)
        # Get face embedding vector via facenet model
        scaledx = np.array(resized)
        scaledx = scaledx.reshape(-1, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE, 3)
        embedding = utils.get_embedding(facenet, scaledx[0])

        # Produce padded crop that will be saved to disk (shown during annotation)
        padded_box = bbox_float_to_int(filtered_box, d_width, d_height, padding=SAVE_FACE_PADDING)
        padded_img = img.crop(tuple(padded_box))
        padded_img.thumbnail((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)

        # Determine if cropped image is actually grayscale. If so, convert.
        padded_a = np.array(padded_img).reshape((-1, 3))
        is_gray = np.all(padded_a[:,0] == padded_a[:,1])
        if is_gray:
            padded_img = ImageOps.grayscale(padded_img)

        # Save face image and features
        # Note: the box is named after the tight crop, even though the saved image
        # uses the padded box
        box_tag = frame_data["tag"] + ":{}_{}_{}_{}".format(*tight_box)
        if save_image:
            padded_img.save(f"{images_dir}/{box_tag}.jpeg", quality=65)
        json.dump({
            "frame": frame_data["index"],
            "tag": box_tag,
            "embedding": embedding.tolist(),
            "box": tight_box,
            "keypoints": face["keypoints"],
            "w": d_width,
            "h": d_height,
        }, features_file, indent=None, separators=(",", ":"))
        features_file.write("\n")

    return len(valid_faces)

def process_frame_new(frame_data, d_width, d_height, features_file, images_dir,
                      min_trajectory_len, save_image):
    """Save faces + features from a frame, and creating face embeddings.
    """
    if debug:
        for face in frame_data["faces"]:
            print(multi_tracker.has_valid_tracker_safe(face["detection_id"]), face)
    # Filter to faces with a valid trajectory (len > MIN)
    valid_faces = [
        face for face in frame_data["faces"]
        if multi_tracker.has_valid_tracker(face["detection_id"])
    ]

    img_np = frame_data["img_np"]
    img = Image.fromarray(img_np)
    for face in valid_faces:
        # Retrieve the posterior bbox filtered by the Kalman filter
        filtered_box = multi_tracker.get_detection_bbox(face["detection_id"])

        # Crop onto face only (tight crop for embedding)
        tight_box = bbox_float_to_int(filtered_box, d_width, d_height)
        det = { 'box': [tight_box[0], tight_box[1],
                        tight_box[2]-tight_box[0], tight_box[3]-tight_box[1]] }
        margin = int(0.1*160)
        cropped = cropBox(img_np, detection=det, margin=margin)
        #print(cropped)
        embeddings = { i : embedders[i].embeddings([cropped])[0].tolist()
                       for i in embedders.keys() }

        # Produce padded crop that will be saved to disk (shown during annotation)
        padded_box = bbox_float_to_int(filtered_box, d_width, d_height, padding=SAVE_FACE_PADDING)
        padded_img = img.crop(tuple(padded_box))
        padded_img.thumbnail((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)

        # Determine if cropped image is actually grayscale. If so, convert.
        padded_a = np.array(padded_img).reshape((-1, 3))
        is_gray = np.all(padded_a[:,0] == padded_a[:,1])
        if is_gray:
            padded_img = ImageOps.grayscale(padded_img)

        # Save face image and features
        # Note: the box is named after the tight crop, even though the saved image
        # uses the padded box
        box_tag = frame_data["tag"] + ":{}_{}_{}_{}".format(*tight_box)
        if save_image:
            padded_img.save(f"{images_dir}/{box_tag}.jpeg", quality=65)
        json.dump({
            "frame": frame_data["index"],
            "tag": box_tag,
            "embeddings": embeddings,
            "box": tight_box,
            "keypoints": face["keypoints"],
            "w": d_width,
            "h": d_height,
        }, features_file, indent=None, separators=(",", ":"))
        features_file.write("\n")

    return len(valid_faces)

def process_video(file, opt: Options):
    """Process entire video and extract face boxes.
    """
    assert opt.shard_i < opt.n_shards and opt.shard_i >= 0, "Bad shard index."

    cap = cv2.VideoCapture(file)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # For invalid video files, cap.get return 0.0. Use that as a validity check here.
    assert n_total_frames > 0, "Invalid video file <"+file+"> cwd="+os.getcwd()

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Attempt to compute display aspect ratio from OpenCV if needed.
    if opt.display_width is None or opt.display_height is None:
        sar = video_w / video_h
        num = cap.get(cv2.CAP_PROP_SAR_NUM)
        den = cap.get(cv2.CAP_PROP_SAR_DEN)
        numerator   = num or 1.0
        denominator = den or 1.0
        par = numerator / denominator
        dar = sar * par
        d_height = video_h
        d_width = round(video_h * dar)
        print('fps={} video_w={} video_h={} sar={} num={} den={} par={} dar={} d_width={}'.
              format(fps, video_w, video_h, sar, num, den, par, dar, d_width))
    else:
        d_height = opt.display_height
        d_width = opt.display_width

    shard_len = (n_total_frames + opt.n_shards - 1) // opt.n_shards
    beg = shard_len * opt.shard_i
    end = min(beg + shard_len, n_total_frames)  # not inclusive
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, beg), \
        f"Couldn't set start frame to: {beg}"

    # We'll write (face) images, features, trajectories and scene changes to disk
    basename_no_ext, _ = os.path.splitext(os.path.basename(file))
    basename_no_ext_split = basename_no_ext.split("-")
    if basename_no_ext_split[0].isdigit(): # MoMaF
        movie_id = int(basename_no_ext_split[0])
    else: # USSEE
        alld = ""
        for zs in basename_no_ext_split:
            if zs.isdigit():
                alld += zs
        movie_id = int(alld)
    
    features_dir = f"{opt.out_path}/{movie_id}-data/features"
    trajectories_dir = f"{opt.out_path}/{movie_id}-data/trajectories"
    scene_changes_dir = f"{opt.out_path}/{movie_id}-data/scene_changes"
    images_dir = f"{opt.out_path}/{movie_id}-data/images"
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(scene_changes_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    features_path = f"{features_dir}/features_{movie_id}_{beg}-{end}.jsonl"
    features_file = open(features_path, mode="w")
    trajectories_path = f"{trajectories_dir}/trajectories_{movie_id}_{beg}-{end}.jsonl"
    trajectories_file = open(trajectories_path, mode="w")
    scene_changes_path = f"{scene_changes_dir}/scene_changes_{movie_id}_{beg}-{end}.json"

    scene = SceneChangeDetector(grayscale=False, crop=True, movie_id=movie_id)
    scene_changes = []

    print(f"Movie file: {os.path.basename(file)}")
    print(f"Total length: {(n_total_frames / fps / 3600):.1f}h ({fps} fps)")
    print(f"Storage resolution for film: {video_w}x{video_h}")
    print(f"Used display resolution for film: {d_width}x{d_height}")
    print(f"Shard {(opt.shard_i + 1)} / {opt.n_shards}, len: {shard_len} frames")
    print(f"Processing frames: {beg} - {end} (max: {n_total_frames}) "
          f"saving every 1/{opt.save_every} frames")

    buf = []
    saved_frames_count = 0
    saved_boxes_count = 0
    saved_traj_count = 0

    # Run past the end to allow easier merging with other shards
    end_with_overlap = min(end + opt.max_trajectory_age, n_total_frames)

    for f in range(beg, end_with_overlap):
        if debug:
            print('I frame', f)
        ret, frame = cap.read()

        # print('frame', f)
        if not ret:
            # print('break')
            break

        frame_h, frame_w, _ = frame.shape
        # If required, resize the frame to display aspect ratio.
        if d_width != video_w or d_height != video_h:
            frame = cv2.resize(frame, (d_width, d_height))

        if f==beg:
            print(frame_h, frame_w, ':', video_h, video_w, ':',
                  d_height, d_width, ':', frame.shape)                  
            
        frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scene_change_happened = scene.update(np.array(frame_img))
        
        faces = detector.detect(frame_img)
        buf.append({
            "index": f,
            "img_np": frame_img,
            "faces": faces,
            "tag": f"{movie_id}:{f}",
        })

        # Stop tracking previous trajectories (faces) if a scene change occurred
        if scene_change_happened:
            scene_changes.append(f)
            multi_tracker.kill_trackers()
            if debug:
                print("scene changed")

        # Let the tracker know of new detections
        detections = np.array([[*f["box"], 0.95] for f in faces]).reshape((-1, 5))
        detection_ids = multi_tracker.update(detections, frame=f)
        # Assign the tracker's detection ids to each face.
        for i, face in enumerate(faces):
            face["detection_id"] = detection_ids[i]

        # Clean up expired trajectories (-> save to file)
        expired_tracks = multi_tracker.pop_expired(2 * opt.min_trajectory, f)
        saved_traj_count += save_trajectories(trajectories_file, expired_tracks, d_width, d_height)

        # For some frames, we save images + features
        # used to include additional condition "and f < end" until 2022-05-19
        if len(buf) == opt.min_trajectory:
            frame_data = buf.pop(0)
            if debug:
                facelist = [ str(ff["detection_id"]) for ff in frame_data["faces"] ]
                print('A frame', frame_data['index'], ', '.join(facelist))
            if frame_data["index"] % opt.save_every == 0:
                n_saved_faces = process_frame_new(
                    frame_data, d_width, d_height, features_file, images_dir,
                    opt.min_trajectory, opt.save_images
                )
                saved_boxes_count += n_saved_faces
                saved_frames_count += int(n_saved_faces > 0)

    # Save remaining frames and trajectories
    for frame_data in buf:
        if debug:
            facelist = [ str(ff["detection_id"]) for ff in frame_data["faces"] ]
            print('B frame', frame_data['index'], facelist)
        if frame_data["index"] % opt.save_every == 0:
            n_saved_faces = process_frame_new(
                frame_data, d_width, d_height, features_file, images_dir,
                opt.min_trajectory, opt.save_images 
            )
            saved_boxes_count += n_saved_faces
            saved_frames_count += int(n_saved_faces > 0)

    expired_tracks = multi_tracker.pop_expired(expiry_age=0)
    saved_traj_count += save_trajectories(trajectories_file, expired_tracks, d_width, d_height)

    # Save scene changes to file
    with open(scene_changes_path, "w") as f:
        scene_changes = [f for f in scene_changes if f >= beg and f < end]
        json.dump({"frame_indices": scene_changes}, f, indent=None, separators=(",", ":"))

    features_file.close()
    trajectories_file.close()
    cap.release()
    print(f"Saved {saved_boxes_count} boxes from {saved_frames_count} different frames")
    print(f"and {saved_traj_count} trajectories.")

if __name__ == "__main__":
    if True or debug:
        print(sys.argv)
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-shards", type=int, default=256, help="sets the number of shards")
    parser.add_argument("--shard-i", type=int, required=True, help="indicates specific shard")
    parser.add_argument("--save-every", type=int, default=5, help="interval between saved frame images")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="required area overlap between matches")
    parser.add_argument("--min-trajectory", type=int, default=3, help="minimum allowed trajectory length")
    parser.add_argument("--max-trajectory-age", type=int, default=5, help="maximum allowed break in trajectory")
    parser.add_argument("--min-face-size", type=int, default=20, help="minimum allowed face size in unknown units")
    parser.add_argument("--face-threshold", type=float, default=0.95, help="minimum allowed face detection score")
    parser.add_argument("--out-path", type=str, default="./data", help="storage directory")
    parser.add_argument("--no-images", action="store_true", help="if set, no images are stored")
    parser.add_argument("file")
    args = parser.parse_args()

    start_time = time()

    # https://stackoverflow.com/questions/67653618/unable-to-load-facenet-keras-h5-model-in-python
    embedders = { i : FaceNet(key=i) for i in facenet_models }

    # Attempt reading precomputed display aspect ratios (DAR) for video files.
    # Not strictly needed but useful for a great result.
    # This is a way to deal with wonky pixel aspect ratios. More info:
    # https://gist.github.com/ekreutz/91f262f96fdf8f20949a27b88f7f4935
    file_basename = os.path.basename(args.file)
    aspects_path = "aspect_ratios.csv"
    display_width = None
    display_height = None
    if os.path.exists(aspects_path):
        csv_file = open(aspects_path, "r")
        reader = csv.reader(csv_file, delimiter=",", quotechar='"')
        headers = next(reader)
        # Required columns: filename, display_width, display_height
        name_i, w_i, h_i = list(map(headers.index, ["filename", "display_width", "display_height"]))
        for row in reader:
            if row[name_i] == file_basename:
                display_width = int(row[w_i])
                display_height = int(row[h_i])
                break
        csv_file.close()

    # Comment out 1, same wrapped api!
    # detector = MTCNNDetector()
    # detector = RetinaFaceDetector(min_face_size=args.min_face_size)
    detector = FaceNetDetector(min_face_size=args.min_face_size,
                               face_threshold=args.face_threshold)

    # Tracker - SORT. Has nothing to do with sorting
    multi_tracker = Sort(
        max_age=args.max_trajectory_age,
        min_hits=args.min_trajectory,
        iou_threshold=args.iou_threshold,
    )

    # Setup options and run extraction process
    options = Options(
        n_shards=args.n_shards,
        shard_i=args.shard_i,
        save_every=args.save_every,
        out_path=args.out_path.rstrip("/"),
        max_trajectory_age=args.max_trajectory_age,
        min_trajectory=args.min_trajectory,
        display_width=display_width,
        display_height=display_height,
        save_images=not args.no_images
    )
    process_video(args.file, options)

    minutes, seconds = divmod(time() - start_time, 60)
    print(f"Completed in {int(minutes)} minutes, {int(seconds)} seconds.")
