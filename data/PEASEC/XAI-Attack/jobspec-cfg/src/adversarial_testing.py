import data.data_reader as data_reader
from modelclass import ModelClass
import logging
import datetime
import os

dir = os.path.dirname(os.path.abspath(__file__))
import argparse

# create parser
parser = argparse.ArgumentParser(
    description="Test adversarial examples on a adversarial dataset."
)
parser.add_argument("--dataset", type=str, help="Dataset name", required=True)
parser.add_argument("--model", type=str, help="Model name", required=True)
parser.add_argument(
    "--wandb_logging",
    action=argparse.BooleanOptionalAction,
    help="Whether to log to wandb",
    default=False,
)

# parse arguments
args = parser.parse_args()
DATASET_NAME = args.dataset
MODEL_NAME = args.model
WANDB_LOGGING = args.wandb_logging

if WANDB_LOGGING:
    import wandb

    # login to wandb with key
    wandb.login(key="KEY")
    wandb.init(
        project="XAI-Attack",
        entity="ENTITY",
        group="adversarial_example_creation",
        name=DATASET_NAME
        + "_"
        + MODEL_NAME
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

logging_path = os.path.join(
    dir,
    "../logs/evaluation/adversarial_testing/" + DATASET_NAME + "/" + MODEL_NAME,
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log",
)
os.makedirs(os.path.dirname(logging_path), exist_ok=True)
# setting up logger to log to file in logs folder with name created from current time and date
logging.basicConfig(
    filename=logging_path,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

x_train, y_train, _, _, x_dev, y_dev = data_reader.read_data(
    DATASET_NAME, adversarial=False
)
num_labels = len(set(y_train))
is_sentence_pair_task = type(x_train) is tuple

# read in the adversarial dataset
x_test, y_test = data_reader.read_data(DATASET_NAME, adversarial=True)

# see how the base model performs on the adversarial dataset
base_model = ModelClass(MODEL_NAME, False, None, num_labels=num_labels)
if is_sentence_pair_task:
    base_model.train(
        (x_train[0] + x_dev[0], x_train[1] + x_dev[1]), y_train + y_dev, x_test, y_test
    )
else:
    base_model.train(x_train + x_dev, y_train + y_dev, x_test, y_test)

# see how a adversarial trained model performs on the adversarial dataset
x_train_adv, y_train_adv = data_reader.read_adv_examples(
    DATASET_NAME, MODEL_NAME, is_sentence_pair_task
)
model_adv = ModelClass(MODEL_NAME, False, None, num_labels=num_labels)
if is_sentence_pair_task:
    model_adv.train(
        (
            x_train[0] + x_train_adv[0] + x_dev[0],
            x_train[1] + x_train_adv[1] + x_dev[1],
        ),
        y_train + y_train_adv + y_dev,
        x_test,
        y_test,
    )
else:
    model_adv.train(
        x_train + x_train_adv + x_dev, y_train + y_train_adv + y_dev, x_test, y_test
    )
