import argparse
from classifier import HotdogClassifier

class AddToDict(argparse.Action):
    """An argparser action that adds arguments to a dictionary with the argument name as key"""
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            return 
        
        if not hasattr(namespace, self.dest):
            setattr(namespace, self.dest, {})
        elif getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        
        key = option_string.replace("-", "") # argument name
        getattr(namespace, self.dest).update({key: values})


parser = argparse.ArgumentParser(description='Hotdog classifier')

# general args
parser.add_argument("-mode", default = "train", choices = ["train", "test"], help = "Mode to run the classifier in")

# classifier args
parser.add_argument("--name", default = None, type = str, help = "Name of the model")
parser.add_argument("--project", default = "Hotdog", type = str, help = "Name of the project")
parser.add_argument('--show-test-images', action='store_true', default=False)
parser.add_argument("--model", type = str, default = None, help = "Model to use for classification")
parser.add_argument("--use_wandb", action='store_true', default=True)

# config args
parser.add_argument("--optimizer", type = str, action=AddToDict, dest = "config", help = "Optimizer")
parser.add_argument("--loss_fun", type = str, action=AddToDict, dest = "config", help = "Loss function")
parser.add_argument("--num_epochs", type = int, action=AddToDict, dest = "config", help = "Batch size")
parser.add_argument("--dropout", type = float, action=AddToDict, dest = "config", help = "Dropout")
parser.add_argument("--batchnorm", type = bool, action=AddToDict, dest = "config", help = "Batchnorm")
parser.add_argument("--finetune", type = bool, action=AddToDict, dest = "config", help = "Finetune")

# optimizer kwargs
parser.add_argument("--lr", type = float, action=AddToDict, dest = "optimizer_kwargs", help = "Learning rate")

# train dataset kwargs
parser.add_argument("--data_augmentation", type = bool, action=AddToDict, dest = "train_dataset_kwargs", help = "Data augmentation")

# parse args
args = parser.parse_args()
args.config = {} if args.config is None else args.config
args.config["optimizer_kwargs"] = {} if args.optimizer_kwargs is None else args.optimizer_kwargs
args.config["train_dataset_kwargs"] = {} if args.train_dataset_kwargs is None else args.train_dataset_kwargs

# create classifier
classifier = HotdogClassifier(name = args.name, project=args.project, show_test_images=args.show_test_images, use_wandb=args.use_wandb, model=args.model, **args.config)

if args.mode == "train":
    classifier.train()
elif args.mode == "test":
    classifier.test()

# print(args)