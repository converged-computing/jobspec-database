from distutils.command.config import config
from flearn.experiments.llama_lora import *
from flearn.utils.options import flArguments
from transformers import HfArgumentParser
import sys
import argparse



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="config_path", required=True, default=None)
    parser.add_argument("--token_num", help="scaffold efficiency", required=False, default=None)
    parser.add_argument("--lr", help="learning rate", required=False, default=None)
    parser.add_argument("--train_batch_size", help="batch_size", required=False, default=None)
    args = parser.parse_args()

    return args




def main():


    args = get_parser()
    # arguments = sys.argv
    # if len(arguments) < 2:
    #     assert len(arguments) == 2, "please input the config file..."

    # config_path = arguments[-1]
    parser = HfArgumentParser((flArguments))
    training_args = parser.parse_json_file(args.path)[0]

    if args.token_num:
        training_args.num_prompt_tokens = int(args.token_num)
    
    if args.lr:
        training_args.learning_rate = float(args.lr)

    if args.train_batch_size:
        training_args.train_batch_size = int(args.train_batch_size)
    t = CentralTraining(training_args)
    t.train()


if __name__ == '__main__':
    main()