import sys 
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='no-rep', help='no-rep=gpt2gen, no-zipfs, has-rep=regular, rm-window-rep')
    parser.add_argument('--task', type=str, default='wp', help='wp, wikitext')

    parser.add_argument('--rand_idx', type=str, default='no',
                        help='no or yes')

    parser.add_argument('--pretrained_model', type=str, default='gpt2', help='')
    parser.add_argument('--model_type', type=str, default='gpt2', help='')

    parser.add_argument('--dataset_name', type=str, default='wikitext', help='')
    parser.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1', help='')
    parser.add_argument('--train_file', type=str, default='wikitext', help='')
    parser.add_argument('--validation_file', type=str, default='wikitext', help='')

    parser.add_argument('--dir_name', type=str, default=None, help='')
    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--block_size', type=int, default=100, help='')

    # training parameters.
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42
    parser.add_argument('--bsz', type=int, default=10, help='')
    parser.add_argument('--epoch', type=int, default=5, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    parser.add_argument('--temperature', type=float, default=1., help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--percent', type=float, default=1.0, help='')

    parser.add_argument('--submit', type=str, default='no', help='')
    parser.add_argument('--use_big', type=str, default='no', help='')

    parser.add_argument('--apply_lora', help='Using LoRA fine-tuning!', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument('--lora_alpha', type=int, default=16, help='')

    parser.add_argument('--threshold_ratio_inhi', type=float, default=0.0, help='')
    parser.add_argument('--gi_key_side', help='Using InA fine-tuning On KEY!', action='store_true')
    parser.add_argument('--gi_query_side', help='Using InA fine-tuning On QUERY!', action='store_true')
    parser.add_argument('--gi_value_side', help='Using InA fine-tuning On VALUE!', action='store_true')

    parser.add_argument('--quant_m', type=str, default='original', help='')
    parser.add_argument('--weight_i_width', type=int, default=0, help='')
    parser.add_argument('--weight_f_width', type=int, default=0, help='')

    parser.add_argument('--app', type=str, default='', help='')


    args = parser.parse_args()
    print('**************************  args  ******************************')
    print(f"arg is: {args}")
    print('**************************  args  ******************************')
    folder_name = "classifier_models"


    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


    if (args.apply_lora is False) and (args.gi_key_side is False) and (args.gi_query_side is False):
        print('Fine-tuning Directly!')
        if args.experiment == 'e2e-tgt' or  args.experiment == 'e2e-tgt-pos' or args.experiment == 'e2e-tgt-tree' or \
                args.experiment == 'e2e-tgt-gen-tree' or  args.experiment == 'e2e-tgt-gen-pos' or args.experiment == 'e2e-back-gen' \
                or args.experiment == 'e2e-tgt-gen-length' or args.experiment == 'e2e-tgt-length' \
                or args.experiment == 'e2e-tgt-gen-spans' or args.experiment == 'e2e-tgt-spans' \
                or args.experiment == 'e2e-back' \
                or args.experiment == 'simple-wiki' or args.experiment == 'roc':

            if args.dataset_name == 'none':
                Model_FILE = args.experiment + '_' + args.quant_m + str(args.weight_i_width) + str(args.weight_f_width) + \
                             '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch, args.bsz * args.gradient_accumulation_steps,
                                                         args.pretrained_model, os.path.basename(args.train_file), args.seed,
                                                               args.task)
                Model_FILE = Model_FILE + f'_{args.notes}'
                logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                Model_FILE = os.path.join(folder_name, Model_FILE)
                app = f" --train_file={args.train_file} --validation_file {args.validation_file} " \
                      f" --task {args.task}"
                app += " " + args.app

            else:
                Model_FILE = args.experiment + '_' + args.quant_m + str(args.weight_i_width) + str(args.weight_f_width) + \
                             '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch, args.bsz * args.gradient_accumulation_steps,
                                                         args.pretrained_model, args.dataset_config_name, args.seed, args.task)
                Model_FILE = Model_FILE + f'_{args.notes}'
                logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                Model_FILE = os.path.join(folder_name, Model_FILE)
                app = f" --dataset_name={args.dataset_name} " \
                      f"--dataset_config_name {args.dataset_config_name} --task {args.task}"
                app += " " + args.app
    else:
        print('Adaption Fine-tuning!')
        if (args.apply_lora is True) and (args.gi_key_side is False) and (args.gi_query_side is False):
            if args.experiment == 'e2e-tgt' or args.experiment == 'e2e-tgt-pos' or args.experiment == 'e2e-tgt-tree' or \
                    args.experiment == 'e2e-tgt-gen-tree' or args.experiment == 'e2e-tgt-gen-pos' or args.experiment == 'e2e-back-gen' \
                    or args.experiment == 'e2e-tgt-gen-length' or args.experiment == 'e2e-tgt-length' \
                    or args.experiment == 'e2e-tgt-gen-spans' or args.experiment == 'e2e-tgt-spans' \
                    or args.experiment == 'e2e-back' \
                    or args.experiment == 'simple-wiki' or args.experiment == 'roc':

                if args.dataset_name == 'none':
                    Model_FILE = args.experiment + '_' + 'LoRA' + '_' + args.quant_m + str(args.weight_i_width) + str(
                        args.weight_f_width) + \
                                 '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch,
                                                                   args.bsz * args.gradient_accumulation_steps,
                                                                   args.pretrained_model,
                                                                   os.path.basename(args.train_file), args.seed,
                                                                   args.task)
                    Model_FILE = Model_FILE + f'_{args.notes}'
                    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                    Model_FILE = os.path.join(folder_name, Model_FILE)
                    app = f" --train_file={args.train_file} --validation_file {args.validation_file} " \
                          f" --task {args.task}"
                    app += " " + args.app

                else:
                    Model_FILE = args.experiment + '_' + 'LoRA' + '_' + args.quant_m + str(args.weight_i_width) + str(
                        args.weight_f_width) + \
                                 '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch,
                                                                   args.bsz * args.gradient_accumulation_steps,
                                                                   args.pretrained_model, args.dataset_config_name,
                                                                   args.seed, args.task)
                    Model_FILE = Model_FILE + f'_{args.notes}'
                    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                    Model_FILE = os.path.join(folder_name, Model_FILE)
                    app = f" --dataset_name={args.dataset_name} " \
                          f"--dataset_config_name {args.dataset_config_name} --task {args.task}"
                    app += " " + args.app


        elif (args.gi_key_side is True) and (args.gi_query_side is True) and (args.apply_lora is False):
            if args.experiment == 'e2e-tgt' or args.experiment == 'e2e-tgt-pos' or args.experiment == 'e2e-tgt-tree' or \
                    args.experiment == 'e2e-tgt-gen-tree' or args.experiment == 'e2e-tgt-gen-pos' or args.experiment == 'e2e-back-gen' \
                    or args.experiment == 'e2e-tgt-gen-length' or args.experiment == 'e2e-tgt-length' \
                    or args.experiment == 'e2e-tgt-gen-spans' or args.experiment == 'e2e-tgt-spans' \
                    or args.experiment == 'e2e-back' \
                    or args.experiment == 'simple-wiki' or args.experiment == 'roc':

                if args.dataset_name == 'none':
                    Model_FILE = args.experiment + '_' + 'InA' + '_' + args.quant_m + str(args.weight_i_width) + str(
                        args.weight_f_width) + \
                                 '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch,
                                                                   args.bsz * args.gradient_accumulation_steps,
                                                                   args.pretrained_model,
                                                                   os.path.basename(args.train_file), args.seed,
                                                                   args.task)
                    Model_FILE = Model_FILE + f'_{args.notes}'
                    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                    Model_FILE = os.path.join(folder_name, Model_FILE)
                    app = f" --train_file={args.train_file} --validation_file {args.validation_file} " \
                          f" --task {args.task}"
                    app += " " + args.app

                else:
                    Model_FILE = args.experiment + '_' + 'InA' + '_' + args.quant_m + str(args.weight_i_width) + str(
                        args.weight_f_width) + \
                                 '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch,
                                                                   args.bsz * args.gradient_accumulation_steps,
                                                                   args.pretrained_model, args.dataset_config_name,
                                                                   args.seed, args.task)
                    Model_FILE = Model_FILE + f'_{args.notes}'
                    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                    Model_FILE = os.path.join(folder_name, Model_FILE)
                    app = f" --dataset_name={args.dataset_name} " \
                          f"--dataset_config_name {args.dataset_config_name} --task {args.task}"
                    app += " " + args.app


        elif (args.apply_lora is True) and (args.gi_key_side is True) and (args.gi_query_side is True):

            print('InA and InA Fine-tuning!')
            if args.experiment == 'e2e-tgt' or args.experiment == 'e2e-tgt-pos' or args.experiment == 'e2e-tgt-tree' or \
                    args.experiment == 'e2e-tgt-gen-tree' or args.experiment == 'e2e-tgt-gen-pos' or args.experiment == 'e2e-back-gen' \
                    or args.experiment == 'e2e-tgt-gen-length' or args.experiment == 'e2e-tgt-length' \
                    or args.experiment == 'e2e-tgt-gen-spans' or args.experiment == 'e2e-tgt-spans' \
                    or args.experiment == 'e2e-back' \
                    or args.experiment == 'simple-wiki' or args.experiment == 'roc':

                if args.dataset_name == 'none':
                    Model_FILE = args.experiment + '_' + 'LoRA_InA' + '_' + args.quant_m + str(args.weight_i_width) + str(
                        args.weight_f_width) + \
                                 '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch,
                                                                   args.bsz * args.gradient_accumulation_steps,
                                                                   args.pretrained_model,
                                                                   os.path.basename(args.train_file), args.seed,
                                                                   args.task)
                    Model_FILE = Model_FILE + f'_{args.notes}'
                    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                    Model_FILE = os.path.join(folder_name, Model_FILE)
                    app = f" --train_file={args.train_file} --validation_file {args.validation_file} " \
                          f" --task {args.task}"
                    app += " " + args.app

                else:
                    Model_FILE = args.experiment + '_' + 'LoRA_InA' + '_' + args.quant_m + str(args.weight_i_width) + str(
                        args.weight_f_width) + \
                                 '_e={}_b={}_m={}_{}_{}_{}'.format(args.epoch,
                                                                   args.bsz * args.gradient_accumulation_steps,
                                                                   args.pretrained_model, args.dataset_config_name,
                                                                   args.seed, args.task)
                    Model_FILE = Model_FILE + f'_{args.notes}'
                    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
                    Model_FILE = os.path.join(folder_name, Model_FILE)
                    app = f" --dataset_name={args.dataset_name} " \
                          f"--dataset_config_name {args.dataset_config_name} --task {args.task}"
                    app += " " + args.app

    COMMANDLINE = f"python transformers/examples/pytorch/language-modeling/run_clm.py \
            --output_dir={Model_FILE} \
            --model_name_or_path={args.pretrained_model} \
            --tokenizer_name={args.pretrained_model} \
            --per_device_train_batch_size {args.bsz} \
            --per_device_eval_batch_size {args.bsz} \
            --save_steps 50000 \
            --num_train_epochs {args.epoch} \
            --do_train --eval_steps 10000 --evaluation_strategy steps \
            --do_eval --dataloader_num_workers 1 \
            --threshold_ratio_inhi {args.threshold_ratio_inhi} \
            --gi_key_side {args.gi_key_side} \
            --gi_query_side {args.gi_query_side} \
            --gi_value_side {args.gi_value_side} \
            --quant_m {args.quant_m} \
            --weight_i_width {args.weight_i_width} \
            --weight_f_width {args.weight_f_width} \
            --apply_lora {args.apply_lora} \
            --lora_r {args.lora_r} \
            --lora_alpha {args.lora_alpha} \
            --save_total_limit 1 \
            --overwrite_output_dir  \
            --logging_dir {logging_dir} \
            --block_size {args.block_size}  \
            --disable_tqdm True --model_type {args.model_type} \
            --gradient_accumulation_steps {args.gradient_accumulation_steps} " \
                  f"--experiment {args.experiment} --seed {args.seed}"


    COMMANDLINE += app

    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)

    print(COMMANDLINE)
    if args.submit == 'no':
        os.system(COMMANDLINE)  # textattack/roberta-base-ag-news # textattack/roberta-base-imdb
