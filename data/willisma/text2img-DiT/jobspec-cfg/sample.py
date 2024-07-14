# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from einops import rearrange
import argparse
import os

from transformers import DistilBertTokenizerFast, DistilBertModel
from coco import CocoDataset, collate_fn
from sample_ddp import center_crop_arr

def ids_to_tokens(tokenizer, cap):
    anns = []
    for i, ids in enumerate(cap):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        ann = ''
        for token in tokens:
            if(token == '[CLS]'):
                continue
            elif(token == '[SEP]'):
                break
            elif(token == ',' or token == '.'):
                ann += token
            elif(token.find('#') != -1):
                token = token.replace('#','')
                ann += token
            else:
                ann += " " + token 

        anns.append(ann[1:])
    return anns

def text_encoding(caption, encoder, end_token):
    device = caption.device
    mask = torch.cumsum((caption == end_token), 1).to(device)
    mask[caption == end_token] = 0
    mask = (~mask.bool()).long()

    emb = encoder(caption, attention_mask=mask)['last_hidden_state']
    
    emb = rearrange(emb, 'b c h -> b (c h)')
    return emb

def update_model_state(model_state, state_dict_path):

    pretrained_state = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
    ckpt_steps = 0

    if "ema" in pretrained_state:
        pretrained_state = pretrained_state["ema"]

    for name, param in pretrained_state.items():
        if name not in model_state:
                continue
        elif isinstance(param, torch.nn.parameter.Parameter):
            param = param.data

        model_state[name].copy_(param)
        
        # if fine_tuning == True and "adaLN_modulation" not in name:
        #     model_state[name].requires_grad = False
    
    return model_state

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    assert os.path.exists(args.data_path), f'Could not find COCO2017 at {args.data_path}'

    val_path = os.path.join(args.data_path, "val2017")
    ann_path = os.path.join(args.data_path, "annotations/captions_val2017.json")

    os.makedirs(args.save_dir, exist_ok=True)

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.seed}"
    sample_folder_dir = f"{args.save_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        emb_dropout_prob=0.0,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    # state_dict = find_model(ckpt_path)
    state_dict = update_model_state(model.state_dict(), ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = CocoDataset(val_path, ann_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    img_idx = 0
    cap_idx = 0
    for _ in range(args.batch):
        # Create text conditioning
        _, cap = next(iter(dataloader))

        # Create sampling noise:
        # n = len(class_labels)
        n = cap.shape[0]
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        # y = torch.tensor(class_labels, device=device)
        y = cap.to(device)
        y = text_encoding(y, encoder, 102)

        # Setup classifier-free guidance:
        # z = torch.cat([z, z], 0)
        # y_null = torch.zeros_like(y).to(device)
        # y = torch.cat([y, y_null], 0)
        # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale) # --> param for penalize unconditional output
        model_kwargs = dict(y=y)

        # Sample images:
        # samples = diffusion.p_sample_loop(
        #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        # )
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        for sample in samples:
            save_image(sample, os.path.join(sample_folder_dir, f"sample-{img_idx}.png"), normalize=True, value_range=(-1,1))
            img_idx += 1

        # save_image(samples, f"sample-{model_string_name}.png", nrow=4, normalize=True, value_range=(-1, 1))
    
        anns = ids_to_tokens(tokenizer, cap)

        with open(os.path.join(args.save_dir, f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                    f"cfg-{args.cfg_scale}-seed-{args.seed}.txt"), 'a') as f:
            for ann in anns:
                f.write(f'{cap_idx}: {ann}\n')
                cap_idx += 1
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="display_samples")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
