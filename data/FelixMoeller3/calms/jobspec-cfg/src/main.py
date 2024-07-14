from utils import config
import argparse

parser = argparse.ArgumentParser(description='Run experiment for model stealing')
parser.add_argument("-c","--config",type=str,help="The location of the config file which will be run")
parser.add_argument("-m","--mode",type=str,help="Which mode shall be run. Can be one of [AL,CL,MS,TR]" \
    "for active learning, continual and active learning, model stealing and target model training respectively")
args = parser.parse_args()
if args.mode == "CL":
    config.run_cl_al_config(args.config)
elif args.mode == "MS":
    config.run_config(args.config)
elif args.mode == "TR":
    config.run_target_model_config(args.config)
else:
    raise ValueError(f"Unknown run mode: {args.mode}. Mode must be one of CL,MS,TR")
