import argparse as ap
import sys
import pickle
import pathlib
import os
import torch

# used for logging. refer to:
# https://docs.wandb.ai/guides/integrations/huggingface
# import wandb

os.environ["WANDB_DISABLED"] = "true"

def run():
    parser = ap.ArgumentParser(description="Use this file with the required command line arguments to train a model and log metrics to wandb.ai")

    parser.add_argument("--output_folder", dest="output_folder", help="path to output folder (will be overwritten if already exists)")

    parser.add_argument("--test_set", dest="test_set", help="path to pickle file with test set")
    parser.add_argument("--valid_set", dest="valid_set", help="path to pickle file with validation set")
    parser.add_argument("--train_set", dest="train_set", help="path to pickle file with training set")

    parser.add_argument("--model_name", dest="model_name", help="name of model to use") # choices=["roberta", "longformer"]

    parser.add_argument("--random_seed", dest="random_seed", type=int, required=True, help="random seed for reproducibility")

    # # hyperparameters
    parser.add_argument("--max_len", dest="max_len", type=int, required=True, help="max length for tokenization")
    parser.add_argument("--batch_size", dest="batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, required=True, help="number of training epochs")
    # parser.add_argument("--adam_beta1", dest="adam_beta1", type=float, default=0.85, help="Adam optimizer beta1") 
    # parser.add_argument("--adam_beta2", dest="adam_beta2", type=float, default=0.999, help="Adam optimizer beta2")
    # parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float, default=1e-8, help="Adam optimizer epsilon")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--dropout", dest="dropout", type=float, default=0, help="dropout")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default=0.3, help="weight decay")
    # parser.add_argument("--pos_weight", dest="pos_weight", type=float, default=None, help="loss weight factor for positive labels")
    parser.add_argument("--num_labels", dest="num_labels", type=int, default=5, help="number of labels in multilabel task")
    parser.add_argument("--context_length", dest="context_length", help="Length of the context")
    parser.add_argument("--eval_frequency", dest="eval_frequency", type=int, default=25, help="number of steps between successive evaluations.")
    parser.add_argument("--model_num", dest="model_num", type=int, default=0, help="Used when multiple models are running at the same time.")

    parser.add_argument("--aux_weight_train", dest="aux_weight_train", type=float, default=0, help="Auxiliary loss weight in training.")
    parser.add_argument("--aux_weight_eval", dest="aux_weight_eval", type=float, default=0, help="How much prediction from auxiliary classifiers is used")
    parser.add_argument("--gold_file", dest="gold_file", help="Gold file name")

    args = parser.parse_args()
    
    ### Once args are parsed successfully, run the more expensive imports and start training
    ### This saves time when there is an error in the command line arguments
    
    # allow imports from a subdirectory
    sys.path.append("./models/")

    # models
    import roberta_for_chunk, roberta_for_chunk_crf, roberta_for_chunk_hier, roberta_for_chunk_hier1, run_existing_model, roberta_for_chunk_hier2
    # helper classes/functions
    from datasets import dataset_dict, Dataset
    
    # columns_to_keep = ['article_id', 'window_start_index', f'{args.context_length}_chunk_tokens', f'labels_in_{args.context_length}', "labels", "bop_index",
    #     'attention_mask', 'labels_mask', 'context_len_for_prediction', 'start_in_context', 'end_in_context', 'min_two_sides']


    train = pickle.load(open(args.train_set, 'rb'))
    train = Dataset.from_pandas(train)
    valid = pickle.load(open(args.valid_set, 'rb'))
    valid = Dataset.from_pandas(valid)

    test = pickle.load(open(args.test_set, 'rb'))
    test = Dataset.from_pandas(test)
    dataset = dataset_dict.DatasetDict({'train': train, 'valid': valid, 'test': test})

    # dataset = dataset.rename_column(f"{args.context_length}_chunk_tokens", "input_ids")
    # dataset = dataset.rename_column(f"labels_in_{args.context_length}", "labels")

    # dataset.set_format(type='torch', columns=[
        # 'article_id', 'labels', 'window_start_index', 'input_ids', 'attention_mask', 'labels_mask',  'context_len_for_prediction', 'start_in_context', 'end_in_context', 'min_two_sides'])
    if args.model_name == 'crf':
        roberta_for_chunk_crf.train( \
        model_name=args.model_name, \
        context_length=args.context_length, \
        batch_size=args.batch_size, \
        num_epochs=args.num_epochs, \
        data_set=dataset, \
        output_dir=args.output_folder, \
        learning_rate=args.learning_rate, \
        model_num=args.model_num, \
        gold_file=args.gold_file, \
        dropout=args.dropout, \
        num_labels=args.num_labels, \
        weight_decay=args.weight_decay, \
        random_seed=args.random_seed, \
        eval_frequency=args.eval_frequency)
    elif args.model_name == 'hier':
        roberta_for_chunk_hier.train( \
            model_name=args.model_name, \
            context_length=args.context_length, \
            batch_size=args.batch_size, \
            num_epochs=args.num_epochs, \
            data_set=dataset, \
            output_dir=args.output_folder, \
            learning_rate=args.learning_rate, \
            model_num=args.model_num, \
            aux_weight_train=args.aux_weight_train, \
            aux_weight_eval=args.aux_weight_eval, \
            gold_file=args.gold_file, \
            dropout=args.dropout, \
            num_labels=args.num_labels, \
            weight_decay=args.weight_decay, \
            random_seed=args.random_seed, \
            eval_frequency=args.eval_frequency)
    elif args.model_name == 'hier1':
        roberta_for_chunk_hier1.train( \
            model_name=args.model_name, \
            context_length=args.context_length, \
            batch_size=args.batch_size, \
            num_epochs=args.num_epochs, \
            data_set=dataset, \
            output_dir=args.output_folder, \
            learning_rate=args.learning_rate, \
            model_num=args.model_num, \
            aux_weight_train=args.aux_weight_train, \
            aux_weight_eval=args.aux_weight_eval, \
            gold_file=args.gold_file, \
            dropout=args.dropout, \
            num_labels=args.num_labels, \
            weight_decay=args.weight_decay, \
            random_seed=args.random_seed, \
            eval_frequency=args.eval_frequency)
    elif args.model_name == 'hier2':
        roberta_for_chunk_hier2.train( \
            model_name=args.model_name, \
            context_length=args.context_length, \
            batch_size=args.batch_size, \
            num_epochs=args.num_epochs, \
            data_set=dataset, \
            output_dir=args.output_folder, \
            learning_rate=args.learning_rate, \
            model_num=args.model_num, \
            gold_file=args.gold_file, \
            dropout=args.dropout, \
            num_labels=args.num_labels, \
            weight_decay=args.weight_decay, \
            random_seed=args.random_seed, \
            eval_frequency=args.eval_frequency)
    elif args.model_name == 'current':
        run_existing_model.train( \
            model_name=args.model_name, \
            context_length=args.context_length, \
            batch_size=args.batch_size, \
            num_epochs=args.num_epochs, \
            data_set=dataset, \
            output_dir=args.output_folder, \
            learning_rate=args.learning_rate, \
            model_num=args.model_num, \
            aux_weight_train=args.aux_weight_train, \
            aux_weight_eval=args.aux_weight_eval, \
            gold_file=args.gold_file, \
            dropout=args.dropout, \
            num_labels=args.num_labels, \
            weight_decay=args.weight_decay, \
            random_seed=args.random_seed, \
            eval_frequency=args.eval_frequency)
    else:
        roberta_for_chunk.train( \
            model_name=args.model_name, \
            context_length=args.context_length, \
            batch_size=args.batch_size, \
            num_epochs=args.num_epochs, \
            data_set=dataset, \
            output_dir=args.output_folder, \
            learning_rate=args.learning_rate, \
            model_num=args.model_num, \
            gold_file=args.gold_file, \
            dropout=args.dropout, \
            num_labels=args.num_labels, \
            weight_decay=args.weight_decay, \
            random_seed=args.random_seed, \
            eval_frequency=args.eval_frequency)
if __name__ == '__main__':
    run()