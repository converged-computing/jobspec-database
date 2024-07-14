import warnings
warnings.filterwarnings('ignore')


import os
import signal
from shutil import copy
from torch.utils.tensorboard import SummaryWriter
from train import Train
from test import Test
from write import PrintRun
import yaml


config_filename = "config.yaml"


if __name__ == "__main__":
    with open(config_filename, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    aug_name = {True: "with",False: "without"}
    fit_name = {0: "",1: "",2:"PenTip_",3:"BothTips_"}
    loader_name = {0:"FeaturesOf",1:"Projections2DOf",2:"CenterPadded",3:"RawDataOf",4:"WithOrientation",5:"OrientedFeaturesOf"}
    name = f'{loader_name[config["dataset"]["loader"]]}_{fit_name[config["dataset"]["fit"]]}{config["dataset"]["name"]}_{aug_name[config["dataset"]["augmentation"]]}_Aug_{config["model"]}'
    out_dir = os.path.join(config["paths"]["output_dir"], name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    copy(f'./{config_filename}',os.path.join(out_dir,config_filename))
    tb_log_dir = os.path.join(out_dir, f"TensorboardLogs_{name}")
    if not os.path.isdir(tb_log_dir):
        os.makedirs(tb_log_dir)
    tensorboard_writer = SummaryWriter(tb_log_dir, comment=name)
    checkpoints_dir = os.path.join(out_dir, f"Checkpoints_{name}")
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    logs_dir = os.path.join(out_dir, f"Logs_{name}")
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
    printer = PrintRun(name)

    def finish_run(sig=None, frame=None):
        exit(0)

    def finish_testing(sig=None, frame=None):
        printer.test_end()
        signal.signal(signal.SIGINT, finish_testing)
        exit(0)

    def finish_training(sig=None, frame=None):
        # tensorboard_writer.close()
        printer.train_end()
        signal.signal(signal.SIGINT, finish_testing)
        printer.test_start()
        Test(
            config=config,
            name=name,
            checkpoints_dir=checkpoints_dir,
            logs_dir=logs_dir
        )
        finish_testing()
        exit(0)

    signal.signal(signal.SIGINT, finish_training)
    printer.train_start()
    Train(
        config=config,
        name=name,
        writer=tensorboard_writer,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir
    )
    finish_training()
