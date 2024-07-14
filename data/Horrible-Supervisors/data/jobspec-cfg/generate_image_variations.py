import os, pdb, sys, time
import numpy as np
import pandas as pd
import argparse as ap

from PIL import Image
from diffusers import StableDiffusionImageVariationPipeline
from diffusers import UnCLIPImageVariationPipeline
from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision import transforms

import torch

import tensorflow as tf
import tensorflow_datasets as tfds

repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_dir)
import data_saver

# data_utils.preprocess_image(image, height, width, is_training=False, color_jitter_strength=0., test_crop=True)
DEVICE = "cuda"
NUM_IMAGES = 10

WIDTH = 224
HEIGHT = 224
CROP_PROPORTION = 0.875  # Standard for ImageNet.

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# tform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# sd_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
# )
# sd_pipe = sd_pipe.to(DEVICE)
# sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

tform = transforms.Compose([
    transforms.ToTensor(),
])
sd_pipe = UnCLIPImageVariationPipeline.from_pretrained(
    "fusing/karlo-image-variations-diffusers", torch_dtype=torch.float16
)
sd_pipe = sd_pipe.to(DEVICE)
sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# tform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(
#         (224, 224),
#         interpolation=transforms.InterpolationMode.BICUBIC,
#         antialias=False,
#         ),
#     transforms.Normalize(
#     [0.48145466, 0.4578275, 0.40821073],
#     [0.26862954, 0.26130258, 0.27577711]),
# ])
# sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
#     "lambdalabs/sd-image-variations-diffusers",
#     revision="v2.0"
# )
# sd_pipe = sd_pipe.to(DEVICE)
# sd_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

shard_lengths = np.array([
    592, 592, 591, 592, 592, 592, 592, 591,
    592, 592, 592, 592, 592, 591, 592, 592
])

def generate_image_variation(img, **kwargs):
    guidance_scale = kwargs.get('guidance_scale', 3.0)
    num_inference_steps = kwargs.get('num_inference_steps', 50)
    inp = img.to(DEVICE).unsqueeze(0)
    # inp = tform(img).to(DEVICE).unsqueeze(0)
    out = sd_pipe(inp, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=NUM_IMAGES)
    return out["images"]

def add_variation_sequence(inp_filepath, out_filepath, **kwargs):
    input_id = kwargs.get('input_id', [-1])
    guidance_scale = kwargs.get('guidance_scale', 3.0)
    num_inference_steps = kwargs.get('num_inference_steps', 50)
    use_range = kwargs.get("use_range", False)
    half = kwargs.get("half", False)

    if use_range and len(input_id) == 2:
        input_id_list = np.arange(input_id[0], input_id[1]).tolist()
    else:
        input_id_list = input_id

    count = 0
    size = 0
    ds = tf.data.TFRecordDataset(inp_filepath)
    for _ in ds.as_numpy_iterator():
       size += 1
    print(f"Size: {size}", flush=True)

    selected_classes = np.arange(1001)
    if half:
        df = pd.read_csv("train-50-1000.csv")
        selected_classes = np.unique(df['label'])
    print(selected_classes)

    start_output_filepath = out_filepath
    inp_image_list = []
    loop_time = time.time()
    last_size = 0
    for element in ds.as_numpy_iterator():
        example = tf.train.Example()
        example.ParseFromString(element)

        cur_id = example.features.feature['id'].int64_list.value[0]
        cur_label = example.features.feature['label'].int64_list.value[0]

        out_filepath = start_output_filepath + f"_{cur_id:06d}"
        print(f"{time.time() - loop_time}, {out_filepath}", flush=True)
        loop_time = time.time()

        # Don't redo samples we have already done.
        if os.path.exists(out_filepath):
            continue

        # Pick only the selected classes
        if cur_label not in selected_classes:
            continue

        if cur_id not in input_id_list:
            continue

        inp_image_bytes = example.features.feature['image'].bytes_list.value[0]
        inp_image = Image.fromarray(np.asarray(tf.image.decode_jpeg(inp_image_bytes, channels=3)))
        # inp_image = tform(Image.fromarray(np.asarray(tf.image.decode_jpeg(inp_image_bytes, channels=3)))).unsqueeze(0)
        # inp_image_list.append(inp_image)
        # inp_image_tensor = torch.cat(inp_image_list)
        # inp = inp_image_tensor.to(DEVICE)
        # out = sd_pipe(inp, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=NUM_IMAGES)
        # out = sd_pipe(inp_image, num_images_per_prompt=NUM_IMAGES)
        out = sd_pipe(inp_image, num_images_per_prompt=NUM_IMAGES, decoder_num_inference_steps=25, decoder_guidance_scale=3.0, super_res_num_inference_steps=10)
        out_image_list = out['images']
        out_image_list_tensors = [tf.convert_to_tensor(x) for x in out_image_list]
        out_images = []
        for cur_image in out_image_list_tensors:
            out_img = data_saver.center_crop(cur_image, WIDTH, HEIGHT, CROP_PROPORTION)
            out_img_pil = tf.keras.preprocessing.image.array_to_img(out_img)
            out_img_tensor = tf.io.encode_jpeg(tf.keras.preprocessing.image.array_to_img(out_img_pil))
            out_images.append(out_img_tensor)
        data_saver.save_variation3(cur_id, cur_label, out_filepath, inp_image_bytes, *out_images)

        # Code for testing output was saved correctly.
        ##############################################
        # ds2 = tf.data.TFRecordDataset(out_filepath)
        # for test_element in ds.as_numpy_iterator():
        #     test_ex = tf.train.Example()
        #     test_ex.ParseFromString(test_element)
        #     out_image_bytes = test_ex.features.feature['image'].bytes_list.value[0]

def main(tfrecord_filepath, out_path, input_id, **kwargs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    use_range = kwargs.get('use_range', False)
    half = kwargs.get('half', False)
    add_variation_sequence(tfrecord_filepath, out_path, input_id=input_id, use_range=use_range, half=half)
    print("Complete", flush=True)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--tfrecord_filepath', '-tfp', type=str, help='Path to a tensorflow record.')
    parser.add_argument('--out_path', '-o', type=str, help='Output path.')
    parser.add_argument('--input_id', '-id', type=int, required=False, nargs='+', help='Input Id.')
    parser.add_argument('--use_range', action='store_true', default=False, help='Range')
    parser.add_argument('--half', action='store_true', default=False, help='50 classes')
    args = parser.parse_args()

    args, _ = parser.parse_known_args()
    kwargs = dict(args._get_kwargs())
    main(**kwargs)
    # /home/jrick6/tensorflow_datasets/imagenette_id/full-size-v2/1.0.0/imagenette_id-train.tfrecord-00000-of-00016