import numpy as np
from sklearn.metrics import f1_score
import random
from lime.lime_text import LimeTextExplainer
import data.data_reader as data_reader
from modelclass import ModelClass
import os

dir = os.path.dirname(os.path.abspath(__file__))
import tqdm
import datetime
import logging
import argparse

# create main parser
parser = argparse.ArgumentParser(
    description="Create adversarial examples for a given model and dataset."
)
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument(
    "--model", type=str, help="Model name", default="distilbert-base-uncased"
)
parser.add_argument(
    "--wandb_logging",
    action=argparse.BooleanOptionalAction,
    help="Whether to log to wandb",
    default=False,
)
parser.add_argument(
    "--filtering",
    type=str,
    help="Filtering method. `none`, `count` or `indicator_words`",
    default="none",
)
parser.add_argument(
    "--random_insertion",
    action=argparse.BooleanOptionalAction,
    help="Whether to use random insertion instead of prefixing of the adversarial words",
    default=False,
)

# parse arguments
args = parser.parse_args()
DATASET_NAME = args.dataset
MODEL_NAME = args.model
WANDB_LOGGING = args.wandb_logging
# set FILTERING to args.filtering if it is set, otherwise to "none"
FILTERING = args.filtering
RANDOM_INSERTION = args.random_insertion

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
    "../logs/main/" + DATASET_NAME + "/" + MODEL_NAME,
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


def subsample_wrong_predicted_texts(
    wrong_predicted_texts, right_labels, predicted_labels, size=-1
):
    combined = list(zip(wrong_predicted_texts, right_labels, predicted_labels))
    random.shuffle(combined)
    wrong_predicted_texts, right_labels, predicted_labels = zip(*combined)

    if size > 0:
        wrong_predicted_texts = wrong_predicted_texts[:size]
        right_labels = right_labels[:size]
        predicted_labels = predicted_labels[:size]

    return wrong_predicted_texts, right_labels, predicted_labels


def get_potential_wrong_class_words(
    model,
    wrong_predicted_texts,
    right_labels,
    predicted_labels,
    all_labels,
    is_sentence_pair_task,
):
    potential_wrong_class_words = []
    logging.info(
        "Explain wrong predictions and extract words that might be responsible for the wrong prediction..."
    )
    for i, wrong_text in enumerate(
        tqdm.tqdm(wrong_predicted_texts, desc="Explaining wrong predictions")
    ):
        explainer = LimeTextExplainer()
        if is_sentence_pair_task:
            input = wrong_text[0] + " _SEP_ " + wrong_text[1]
        else:
            input = wrong_text

        if len(all_labels) == 2:
            explain_result = explainer.explain_instance(
                input,
                model.predictor,
                num_features=15,
                num_samples=500,
                is_sentence_pair_task=is_sentence_pair_task,
            )

            explain_result_list = explain_result.as_list()
            # gather all words that might be responsible to distort the label:
            # If the label is 0, the words that have a negative weight are responsible for the wrong prediction
            # If the label is 1, the words that have a positive weight are responsible for the wrong prediction
            wrong_class_words = [
                (word, value)
                for word, value in explain_result_list
                if (right_labels[i] == 1 and value < 0)
                or (right_labels[i] == 0 and value > 0)
            ]
            if len(wrong_class_words) != 0:
                potential_wrong_class_words.append(
                    (right_labels[i], predicted_labels[i], wrong_class_words[0][0])
                )

        if len(all_labels) > 2:
            explain_result = explainer.explain_instance(
                input,
                model.predictor,
                num_features=15,
                num_samples=500,
                labels=all_labels,
                is_sentence_pair_task=is_sentence_pair_task,
            )

            for label in all_labels:
                explain_result_list = explain_result.as_list(label=label)

                wrong_class_words = [
                    (word, value)
                    for word, value in explain_result_list
                    if (right_labels[i] == label and value < 0)
                ]
                if len(wrong_class_words) != 0:
                    potential_wrong_class_words.append(
                        (right_labels[i], predicted_labels[i], wrong_class_words[0][0])
                    )

    potential_wrong_class_words = set(potential_wrong_class_words)
    return potential_wrong_class_words


### Indicator word filtering
def get_potential_right_class_words(
    model, right_predicted_texts, right_labels, all_labels, is_sentence_pair_task
):
    potential_right_class_words = []
    logging.info(
        "Explain right predictions and extract words that might be responsible for the right prediction..."
    )
    for i, right_text in enumerate(
        tqdm.tqdm(right_predicted_texts, desc="Explaining right predictions")
    ):
        explainer = LimeTextExplainer()
        if is_sentence_pair_task:
            input = right_text[0] + " _SEP_ " + right_text[1]
        else:
            input = right_text

        if len(all_labels) == 2:
            explain_result = explainer.explain_instance(
                input,
                model.predictor,
                num_features=15,
                num_samples=500,
                is_sentence_pair_task=is_sentence_pair_task,
            )

            explain_result_list = explain_result.as_list()
            # gather all words that might be responsible to distort the label:
            # If the label is 0, the words that have a negative weight are responsible for the right prediction
            # If the label is 1, the words that have a positive weight are responsible for the right prediction
            right_class_words = [
                (word, value)
                for word, value in explain_result_list
                if (right_labels[i] == 1 and value > 0)
                or (right_labels[i] == 0 and value < 0)
            ]
            if len(right_class_words) != 0:
                potential_right_class_words.append(
                    (right_labels[i], right_labels[i], right_class_words[0][0])
                )

        if len(all_labels) > 2:
            explain_result = explainer.explain_instance(
                input,
                model.predictor,
                num_features=15,
                num_samples=500,
                labels=all_labels,
                is_sentence_pair_task=is_sentence_pair_task,
            )

            for label in all_labels:
                explain_result_list = explain_result.as_list(label=label)

                right_class_words = [
                    (word, value)
                    for word, value in explain_result_list
                    if (right_labels[i] == label and value > 0)
                ]
                if len(right_class_words) != 0:
                    potential_right_class_words.append(
                        (right_labels[i], right_labels[i], right_class_words[0][0])
                    )

    # only keep words with certain frequency
    potential_right_class_words_with_frequency = []
    for label, predicted_label, word in set(potential_right_class_words):
        if (
            potential_right_class_words.count((label, predicted_label, word))
            > right_labels.count(label) * 0.01
        ):
            potential_right_class_words_with_frequency.append(
                (label, predicted_label, word)
            )

    return potential_right_class_words_with_frequency


def random_insert_word(word, text):
    words = text.split()
    position = random.randint(0, len(words))
    words.insert(position, word)
    return " ".join(words)


def create_adversarial_examples(
    model,
    potential_wrong_class_words,
    texts_of_interest,
    max_changes_per_word,
    is_sentence_pair_task,
    label,
):
    result = []
    adversarial_examples = []
    logging.info("Creating adversarial examples for each potential wrong class word...")
    for label_for_word, predicted_label, word in tqdm.tqdm(
        potential_wrong_class_words, desc="Creating adversarial examples"
    ):
        if label_for_word != label:
            continue
        if is_sentence_pair_task:
            if RANDOM_INSERTION:
                instances_of_interest_altered = [
                    (random_insert_word(word, tuple[0]), tuple[1])
                    for tuple in texts_of_interest
                ]
                instances_of_interest_altered += [
                    (tuple[0], random_insert_word(word, tuple[1]))
                    for tuple in texts_of_interest
                ]
            else:
                instances_of_interest_altered = [
                    (word + " " + tuple[0], tuple[1]) for tuple in texts_of_interest
                ]
                instances_of_interest_altered += [
                    (tuple[0], word + " " + tuple[1]) for tuple in texts_of_interest
                ]
        else:
            if RANDOM_INSERTION:
                instances_of_interest_altered = [
                    random_insert_word(word, text) for text in texts_of_interest
                ]
            else:
                instances_of_interest_altered = [
                    word + " " + text for text in texts_of_interest
                ]

        predictions_of_altered_texts = model.predict(instances_of_interest_altered)

        changed_predictions = np.where(predictions_of_altered_texts != label)[0]

        # create adversarial examples: (label, instance)
        # taking those altered instances where the word did change the true prediction but also not too often
        if (
            len(changed_predictions) < max_changes_per_word
            and len(changed_predictions) > 0
        ):
            result.append((label, word, len(changed_predictions)))
            adversarial_examples.extend(
                [instances_of_interest_altered[index] for index in changed_predictions]
            )

    return adversarial_examples, result


x_train, y_train, x_test, y_test, x_dev, y_dev = data_reader.read_data(
    DATASET_NAME, adversarial=False
)
all_labels = list(set(y_train))
is_sentence_pair_task = type(x_train) is tuple


if len(x_dev) == 0:
    x_train, y_train, x_dev, y_dev = data_reader.split_train(x_train, y_train)

# check for local model
if DATASET_NAME == "mnli_mismatched" or DATASET_NAME == "mnli_matched":
    __dataset_name = "mnli"
else:
    __dataset_name = DATASET_NAME
if local := os.path.exists(
    os.path.join(dir, "../model/" + __dataset_name + "/" + MODEL_NAME)
):
    model_path = os.path.join(dir, "../model/" + __dataset_name + "/" + MODEL_NAME)
else:
    model_path = MODEL_NAME

model = ModelClass(MODEL_NAME, local, model_path, len(all_labels))
if not local:
    model.train(x_train, y_train, x_dev, y_dev)
    model.save(os.path.join(dir, "../model/" + __dataset_name + "/" + MODEL_NAME))
logging.info("Model loaded")
predictions = model.predict(x_dev)
# log the f1 score
logging.info("[SANITY CHECK] F1-Score on dev set")
logging.info(f1_score(y_dev, predictions, average="macro"))
if WANDB_LOGGING:
    wandb.log({"F1-Score on dev set": f1_score(y_dev, predictions, average="macro")})

# extract texts of the wrong predictions
right_predictions = np.equal(predictions, y_dev)
wrong_predicted_texts, right_labels, predicted_labels = [], [], []
for i, prediction_is_right in enumerate(right_predictions):
    if not prediction_is_right:
        if is_sentence_pair_task:
            # change of structure: wrong_predicted_texts: [(x0_1, x1_1), (x0_2, x1_2), ..]
            # x_train: x_dev: ([x0_1, x0_2 ..], [x1_1, x1_2, ..])
            wrong_predicted_texts.append((x_dev[0][i], x_dev[1][i]))
        else:
            wrong_predicted_texts.append(x_dev[i])

        right_labels.append(y_dev[i])
        predicted_labels.append(predictions[i])


wrong_predicted_texts, right_labels, predicted_labels = subsample_wrong_predicted_texts(
    wrong_predicted_texts, right_labels, predicted_labels
)

# Tuple-List of structure: (right_label, [words_that_might_distort_label])
potential_wrong_class_words = get_potential_wrong_class_words(
    model,
    wrong_predicted_texts,
    right_labels,
    predicted_labels,
    all_labels,
    is_sentence_pair_task,
)
logging.info("Words that might distort the label of the wrong predicted instances: ")
logging.info(potential_wrong_class_words)
# if WANDB_LOGGING:
#    wandb.log({"Words that might distort the label of the wrong predicted instances": potential_wrong_class_words})


if FILTERING == "indicator_words":
    ### Indicator words filtering
    # extract texts of the right predictions
    right_predictions = np.equal(predictions, y_dev)
    right_predicted_texts, right_labels_for_right_class, predicted_labels = [], [], []
    for i, prediction_is_right in enumerate(right_predictions):
        if prediction_is_right:
            if is_sentence_pair_task:
                # change of structure: wrong_predicted_texts: [(x0_1, x1_1), (x0_2, x1_2), ..]
                # x_train: x_dev: ([x0_1, x0_2 ..], [x1_1, x1_2, ..])
                right_predicted_texts.append((x_dev[0][i], x_dev[1][i]))
            else:
                right_predicted_texts.append(x_dev[i])

            right_labels_for_right_class.append(y_dev[i])

    potential_right_class_words = get_potential_right_class_words(
        model,
        right_predicted_texts,
        right_labels_for_right_class,
        all_labels,
        is_sentence_pair_task,
    )
    logging.info(
        "Words that might distort the label of the wrong predicted instances: "
    )
    logging.info(potential_right_class_words)

    potential_wrong_class_words_list = [
        word for _, _, word in potential_wrong_class_words
    ]
    potential_right_class_words_list = [
        word for _, _, word in potential_right_class_words
    ]
    logging.info(
        "#############################potential wrong class############################"
    )
    logging.info(potential_wrong_class_words)
    logging.info(
        "#############################potential right class############################"
    )
    logging.info(potential_right_class_words)

    matching_words = []
    not_matching_words = []
    potential_wrong_class_words_cleaned = []
    for i, wrong_class_word in enumerate(potential_wrong_class_words_list):
        if wrong_class_word in potential_right_class_words_list:
            matching_words.append(wrong_class_word)
        else:
            not_matching_words.append(wrong_class_word)

    for label, predicted_label, word in potential_wrong_class_words:
        if word in not_matching_words:
            potential_wrong_class_words_cleaned.append((label, predicted_label, word))

    potential_wrong_class_words = potential_wrong_class_words_cleaned

    logging.info(
        "##############################################################################"
    )
    logging.info(
        "#############################non matching words############################"
    )
    logging.info(not_matching_words)
    logging.info(
        "#############################matching words############################"
    )
    logging.info(matching_words)

    ###

## check which words also change the prediction of the instances that were right predicted
instances_right_predicted = np.equal(predictions, y_dev)
result, all_adversarial_examples_dict = [], {}
for label in all_labels:
    instances_of_class_label = np.equal(y_dev, label)
    # instances of interest are the instances that were right predicted and are of the class given by the label
    instances_of_interest = np.logical_and(
        instances_right_predicted, instances_of_class_label
    )

    if is_sentence_pair_task:
        texts_of_interest = [
            tuple
            for i, tuple in enumerate(zip(x_dev[0], x_dev[1]))
            if instances_of_interest[i]
        ]
        max_changes_per_word = len(x_dev[0]) / 30
    else:
        texts_of_interest = [
            text for i, text in enumerate(x_dev) if instances_of_interest[i]
        ]
        max_changes_per_word = len(x_dev) / 30

    if FILTERING != "count":
        max_changes_per_word = float("inf")
    adversarial_examples, res = create_adversarial_examples(
        model,
        potential_wrong_class_words,
        texts_of_interest,
        max_changes_per_word,
        is_sentence_pair_task,
        label,
    )
    result.append(res)
    all_adversarial_examples_dict[label] = adversarial_examples

all_adversarial_examples_list = [
    item for sublist in list(all_adversarial_examples_dict.values()) for item in sublist
]
if is_sentence_pair_task:
    all_adversarial_examples_list = list(map(list, zip(*all_adversarial_examples_list)))
    tmp_predictions = model.predict(
        (all_adversarial_examples_list[0], all_adversarial_examples_list[1])
    )
else:
    tmp_predictions = model.predict(all_adversarial_examples_list)

labels = []
for label in all_labels:
    labels += len(all_adversarial_examples_dict[label]) * [label]

for label in all_labels:
    logging.info(label)
    logging.info(
        sorted(
            [r for r in result if r[0] == label],
            key=lambda tuple: tuple[2],
            reverse=True,
        )
    )

    # if WANDB_LOGGING:
    #    wandb.log({label: sorted([r for r in result if r[0] == label], key=lambda tuple: tuple[2], reverse=True)})

# save the adversarial examples

for label in all_labels:
    if is_sentence_pair_task:
        filepath = os.path.join(
            dir,
            "results/adversarial_examples/"
            + DATASET_NAME
            + "/"
            + MODEL_NAME
            + "/adversarial_examples_class_"
            + str(label)
            + ".txt",
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for adv_ex_tuple in all_adversarial_examples_dict[label]:
                f.write(adv_ex_tuple[0] + "_SEP_" + adv_ex_tuple[1])
                f.write("\n")
    else:
        filepath = os.path.join(
            dir,
            "results/adversarial_examples/"
            + DATASET_NAME
            + "/"
            + MODEL_NAME
            + "/adversarial_examples_class_"
            + str(label)
            + ".txt",
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(all_adversarial_examples_dict[label]))

# save the results
filepath = os.path.join(
    dir,
    "results/adversarial_examples/" + DATASET_NAME + "/" + MODEL_NAME + "/result.txt",
)
os.makedirs(os.path.dirname(filepath), exist_ok=True)
with open(filepath, "w", encoding="utf-8") as f:
    f.write("\n".join([str(r) for r in result]))
