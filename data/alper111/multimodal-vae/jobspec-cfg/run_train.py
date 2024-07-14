import os
import sys

import yaml

sizes = [1, 2, 4, 8, 10, 15, 20, 30, 40]
N = int(sys.argv[1])
for s in sizes:
    file = open(os.path.join("opts-trainjob-%d.yml" % N), "w")
    opts = {}
    opts["save"] = "save/imgjoint-%d-%d" % (s, N)
    opts["data"] = "data"
    opts["device"] = "cuda"
    opts["modality"] = ["img", "joint"]
    opts["action"] = ["grasp", "move"]
    opts["batch_size"] = 128
    opts["epoch"] = 200
    opts["lambda"] = 1.0
    opts["beta"] = 0.0
    opts["init_method"] = "xavier"
    opts["lr"] = 0.0005
    opts["reduce"] = True
    opts["mse"] = True
    opts["beta_decay"] = 0.0
    opts["in_blocks"] = [
        [-2, 1024, 128, 6, 32, 64, 64, 128, 128, 256],
        [-1, 14, 32, 64, 64, 128, 128, 256, 128]
    ]
    opts["in_shared"] = [256, 256]
    opts["out_shared"] = [128, 256]
    opts["out_blocks"] = [
        [-2, 128, 1024, 256, 256, 128, 128, 64, 64, 32],
        [-1, 128, 256, 128, 128, 64, 64, 32, 28]
    ]
    opts["traj_count"] = s
    yaml.dump(opts, file)
    file.close()
    print("Started training with %d trajectories, #%d" % (s, N))
    os.system("python train.py -opts opts-trainjob-%d.yml" % N)
    os.system("python test.py -opts save/imgjoint-%d-%d/opts.yaml -banned 0 0 -prefix both" % (s, N))
    os.system("python test.py -opts save/imgjoint-%d-%d/opts.yaml -banned 0 1 -prefix img" % (s, N))
    os.system("python test.py -opts save/imgjoint-%d-%d/opts.yaml -banned 1 0 -prefix joint" % (s, N))
