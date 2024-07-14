# Stable Diffusion Script
# Accepts a text file as an input, iterates through the file and generates prompts.
# Creates a folder with the ID tag of the image, then generates images inside.
import argparse, os
import sys
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
# GPU Monitoring
import GPUtil
from threading import Thread
import time
import random
# ============================================================================================================================================
# ============================================================================================================================================

# MAIN VARIABLES
torch.set_grad_enabled(False)
random.seed(None)
# directory of negative prompt
neg_dir = "/scratch/jhh508/stable-diffusion-2/negPrompt.txt"
# generate fixed random seeds (5)
# seeds = []
# for i in range(5):
#     tempNum = random.randint(0,1000000)
#     seeds.append(tempNum)
# generate random seeds every time
num_seeds = 1
# steps
steps = [10, 20, 30, 50]
# guidance scales
scales = [7.5, 15.0, 20.0]



# ============================================================================================================================================
# ============================================================================================================================================

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        nargs="?",
        help="the prompt to render"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="/scratch/jhh508/web-diffusion-neg/"
    )
    parser.add_argument(
        "--negFile",
        action='store_true',
        help="dir to negative prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference-v.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="768-v-ema.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        nargs="?",
        default = "",
        help = "the prompt to negate",
    )
    opt = parser.parse_args()
    return opt

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
    
    # def __init__(self, delay):
    #     super(Monitor, self).__init__()
    #     self.stopped = False
    #     self.delay = delay # Time between calls to GPUtil
    #     self.start()
    #     self.loadSum = 0.0
    #     self.timesCounted = 0.0
    #     self.topUsage = 0.0

    # def run(self):
    #     while not self.stopped:
    #         gpu = GPUtil.getGPUs()[0]
    #         # get max load
    #         if(gpu.load > self.topUsage):
    #             self.topUsage = gpu.load
    #         self.loadSum += gpu.load
    #         self.timesCounted += 1.0
    #         time.sleep(self.delay)
    # def stop(self):
    #     self.stopped = True    


def main(opt):

    if(opt.ckpt == "512-base-ema.ckpt"):
        opt.outdir = "/scratch/jhh508/web-diffusion-base-neg/"

    #deviceID = GPUtil.getFirstAvailable()
    # using one GPU atm
    GPUs = GPUtil.getGPUs()
    gpu = GPUs[0]
    totalLoad = 0   # total of loads combined
    totalCount = 0      # times counted

    if(opt.negFile):
        neg_file = open(neg_dir, "r")
        opt.neg_prompt = neg_file.readline()
        neg_file.close()

    # Text File Input
    file = open(opt.csv, "r")

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda")
    
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    if(not opt.negFile):
       opt.outdir = "/scratch/jhh508/web-diffusion/"
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    

    # keep seed and start code the same
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    

    data = []
    ids = []
    for line in file:
        line = line.replace("\"","").strip().replace("data/","").replace(".c","").split(",")
        promptLine = line[1] + ", " + line[2]
        ids.append(line[0])
        data.append(promptLine)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope():

            # iterate through num of images to be produced
            for i in range(num_seeds):

                # set seed each prompt (x seeds per prompt)
                seed = random.randint(0,1000000)
                opt.seed = seed
                # print("seed: " + str(seed))

                seed_everything(opt.seed)

                
                # set sample folder path
                sample_path = os.path.join(outpath, "web-diffusion-images_" + str(seed) + "_" + str(opt.W))
                print(sample_path)
                os.makedirs(sample_path, exist_ok=True)
                sample_count = 0
                base_count = len(os.listdir(sample_path))

                # prompt counter
                counter = 0

                for n in trange(opt.n_iter, desc="Sampling"):
                        
                    # start monitor (GPU track)
                    # monitor = Monitor(0.1)
                    for prompts in tqdm(data, desc="data"):


                        prompt_id = ids[counter]
                        counter += 1

                        # set ID path for image set
                        # id_path = os.path.join(sample_path, prompt_id)
                        # os.makedirs(id_path, exist_ok=True)

                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(opt.neg_prompt)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        print("Prompts: " + prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        # iterate through guidance scales
                        for g_scale in scales:

                            # increment steps, run sampler steps 30, 50, 70
                            for step in steps:
                                print("step ",step)
                                # GPUtil.showUtilization()
                                start_time = time.time()
                                samples, _ = sampler.sample(S=step,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=g_scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)
                                time_taken = (time.time() - start_time)
                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                

                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img_name = f"{prompt_id}" + "_" + f"{step}" + "_" + f"{g_scale}" + ".jpg"
                                    img.save(os.path.join(outpath, img_name))
                                    # img.save(os.path.join(id_path, img_name))
                                    base_count += 1
                                    sample_count += 1
                                print("--- %s seconds ---" % (time.time() - start_time))

                                # put time taken for generating file into log file
                                # writeFile = open(os.path.join(id_path, f"{prompt_id}" + f"log.txt"), "a")
                                writeFile = open(os.path.join(outpath, f"{prompt_id}" + f"_log.txt"), "a")
                                if (step == 10):
                                    writeFile.writelines("Prompt: " + prompts + " \n")
                                    writeFile.writelines("Guidance Scale: " + str(g_scale) + " \n")
                                    writeFile.writelines("Steps, Time Taken \n")
                                writeFile.writelines(str(step) + "," + str(time_taken) + "\n")
                                writeFile.close()
                                
                                # print("Top Usage: " + str(monitor.topUsage) + " AVG: " + str(monitor.loadSum/float(monitor.timesCounted)))
                                # totalLoad += monitor.loadSum
                                # totalCount += monitor.timesCounted
                                # monitor.stop()
                
                            # print("Total AVG Load: " + str(totalLoad/totalCount))
                        # monitor.stop()

    print(f"Finished, outputs in:\n{outpath} \n")

    # stop monitor
    # monitor.stop()

if __name__ == "__main__":
    opt = parse_args()
    main(opt)