"""
python run/main.py
to run with defaults
"""
import torch
import torch.nn as nn
from torch import optim
import random
import wandb
import copy
import os
from functools import partial
from filelock import FileLock
from datetime import datetime
import configargparse as argparse
import json

from transformers import AutoTokenizer, PretrainedConfig, GenerationConfig
from dataset.msc_binary import MSC_Turn_Facts
from models.persona_extractor import PersonaExtractor
from models.bert_classifier import PrefixBert
from models.bart_extractor import PrefixBart, BartExtractor, ExtractedFactLoss
from models.t5_extractor import T5Extractor
from models.dialogpt import DialoGPT
from models.speechact_clf import SpeechactClassifier
from models.knowledge_grounded_generator.kg_model import KnowledgeGroundedDecoder, KG_loss
from models.knowledge_grounded_generator.kg_utils import ConceptGraph
from dataset.msc_kg_sessions import KG_enriched_MSC_Session
from dataset.msc_sessions import MSC_Session
from dataset.convai2 import ConvAI2
from dataset.msc_summary_turns import MSC_Turns
from dataset.msc_summary import MSC_Summaries
from dataset.msc_speechact import MSC_SpeechAct
from dataset.tokenizer import train_tokenizer, Tokenizer, UNK_TOKEN, END_TOKEN, PAD_TOKEN
from metrics.terp import TerpMetric
from metrics.nli import NLIMetric
from run.tune import do_tune
from ray.air import session, RunConfig
from ray.tune import with_resources
from utils.general import savename, prettydict, dict_with_key_prefix, save_config, load_config, save_dict
from utils.listdict import ListDict
import utils.logging as logging


def train(model, trainloader, validloader, optimizer, criterion, 
    device, epochs, log_interval, valid_interval, patience,
    do_tune, use_wandb):

    train_losses = []
    saved_stats = {"valid_loss": float('inf')}
    step = 0
    model.to(device)
    criterion.to(device)
    best_model = model
    num_batches = len(trainloader)
    if patience is None:
        patience = num_batches * epochs
    patience_count = patience
    total_original_tokens, total_truncated_tokens = 0, 0

    for epoch in range(epochs):

        # Train for one epoch
        model.train()
        for batch in iter(trainloader):
            step += 1
            if hasattr(batch, "num_truncated_tokens"):
                total_original_tokens += batch.num_original_tokens.sum().item()
                total_truncated_tokens += batch.num_truncated_tokens.sum().item()

            loss = model.train_step(batch, optimizer, criterion, device)
            train_losses.append(loss)
    
            if step % log_interval == 0:
                loss_avg = sum(train_losses[-log_interval:]) / log_interval
                if use_wandb:
                    wandb.log({"train_loss": loss_avg, "epoch": epoch}, step=step)
                logging.info("Epoch {}, step {}: Train loss={:.4f}".format(epoch, step, loss_avg))    
    
            if (step % valid_interval == 0) or (step % num_batches == 0):
                # Evaluate on validation set
                model.eval()
                valid_stats = valid(model, validloader, criterion, device)
                valid_stats = dict_with_key_prefix(valid_stats, prefix="valid_")
                logging.info("Epoch {}, step {}: Validation stats={}".format(epoch, step, valid_stats))
                patience_count -= 1
                model.train()

                if use_wandb:
                    valid_stats["epoch"] = epoch
                    wandb.log(valid_stats, step=step)

                if do_tune:
                    session.report(valid_stats)

                if valid_stats["valid_loss"] < saved_stats["valid_loss"]:
                        saved_stats = valid_stats
                        logging.info("Best loss improved to {:.4f}".format(saved_stats["valid_loss"]))
                        patience_count = patience
                        best_model = copy.deepcopy(model)

            if patience_count < 0:
                logging.info("Training loop terminated because it ran out of patience after {} validation interval(s) without improvement".format(patience + 1))
                break
        if patience_count < 0: 
            logging.info("Training loop terminated after epoch {}, step {}".format(epoch, step))
            break
    logging.info("Average truncation: {}".format(total_truncated_tokens / max(total_original_tokens, 1)))
    return best_model, saved_stats


def valid(model, dataloader, criterion, device):

    valid_stats = ListDict()
    model.to(device)
    criterion.to(device)
    model.eval()

    for batch in iter(dataloader):

        stats = model.valid_step(batch, criterion, device)
        valid_stats.append(stats)

    stats = valid_stats.mean()

    return stats

def evaluate(model, testdata, args):

    if args.task == "classify":
        eval_kwargs = {'device': args.device}
    elif args.task == "clf_act":
        eval_kwargs = {'device': args.device, 'batch_size': args.batch_size}
    elif args.task == "generate" or args.task == "summarize":
        if args.device == 'mps':
            args.device = 'cpu'
            logging.warning("Changed device from 'mps' to 'cpu' for evaluation")
        TerpMetric.set(terp_dir=args.terpdir, java_home=args.java_home, tmp_dir=args.tmpdir)
        NLIMetric.set(nli_model=args.nli_model, device=args.device, batch_size=args.batch_size)
        eval_kwargs = {
            'generation_config': {
                "num_beams": args.num_beams,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.decoder_max,
            },
            'device': args.device, 
            'log_interval': args.log_interval
        }
        if args.task == 'generate':
            eval_kwargs.update({'batch_size': args.batch_size})
        else:
            eval_kwargs.update({'metrics': args.metrics})
    elif args.task == "dialog":
        if args.device == 'mps':
            args.device = 'cpu'
            logging.warning("Changed device from 'mps' to 'cpu' for evaluation")
        eval_kwargs = {
            'generation_config': GenerationConfig(
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.decoder_max,
            ),
            'device': args.device, 
            'batch_size': 4,
        }
        if args.model == "dialogpt":
            testdata.batch_format = "huggingface_x"

    logging.info("Evaluating model on {} samples of testdata in {} with arguments {}".format(len(testdata), args.basedir, eval_kwargs))
    eval_stats, result_dict = testdata.evaluate(model, **eval_kwargs)
    logging.report(prettydict(eval_stats, title="Eval_stats"))

    return eval_stats, result_dict 

def chat(model, testdata, args):

    if args.device == 'mps':
        args.device = 'cpu'
        logging.warning("Changed device from 'mps' to 'cpu' for chat")

    gen_config = GenerationConfig(
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.decoder_max,
    )

    if args.chat_initfile != "":
        chat_initbatch = []
        with open(args.chat_initfile, 'r') as f:
            for line in f:
                d, t, m = line.split('\t')
                chat_initbatch.append((int(d), int(t), m[:-1]))
            logging.info(f"Loaded {len(chat_initbatch)} chat inits from {args.chat_initfile}")
    else:
        chat_initbatch = [(args.chatdialog_id, args.chatturn_id, args.user_message)]    
        logging.info(f"Continue interactive chat on dialogue {args.chatdialog_id}, turn {args.chatturn_id}")

    stats, chat_results = testdata.chat(model, chat_initbatch, gen_config, device=args.device)
    return stats, chat_results 


def selfchat(models, testdata, args):

    if args.device == 'mps':
        args.device = 'cpu'
        logging.warning("Changed device from 'mps' to 'cpu' for selfchat")

    gen_config = GenerationConfig(
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.decoder_max,
    )
    if not args.new_agent:
        gen_config_other = gen_config
    else:
        gen_config_other = GenerationConfig(
            num_beams=args.num_beams_other,
            do_sample=args.do_sample_other,
            temperature=args.temperature_other,
            top_p=args.top_p_other,
            top_k=args.top_k_other,
            max_new_tokens=args.decoder_max,
        )
    eval_kwargs = {
        'generation_configs': (gen_config, gen_config_other),
        'device': args.device,
        'num_turns': args.num_turns,
    }

    logging.info(f"Performing selfchat on {len(testdata[0])} samples of testdata in {args.basedir} with arguments {eval_kwargs}")
    stats, selfchat_results = MSC_Session.selfchat(models, testdata, **eval_kwargs)
    logging.report(prettydict(stats, title="Selfchat_stats"))

    return stats, selfchat_results 

def prepare_model_and_data(args):

    model, traindata, validdata, testdata, collate_fn, criterion = None, None, None, None, None, None

    if args.task == 'classify' or args.task == 'clf_act': 
        
        num_classes = {'classify': MSC_Turn_Facts.num_classes, 'clf_act': MSC_SpeechAct.num_classes}[args.task]

        # Classify whether dialog turns contain a fact
        if args.model == "bert":

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
                if (args.freeze is not None) and (num_added_toks > 0):
                    logging.warning("Added tokens {} are not trained, because part of model parameters is frozen (freeze={})".format(args.add_tokens, args.freeze))
            model = PrefixBert('bert-base-uncased', freeze=args.freeze, prefix_size=args.prefix_size, prefix_aggr=args.prefix_aggr, num_classes=num_classes)
            model.bert.resize_token_embeddings(len(tokenizer))
            criterion = nn.NLLLoss(reduction='mean')

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        if args.task == 'classify':
            MSC_Turn_Facts.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
            dataset_config = {
                'basedir': args.datadir + args.basedir,
                'sessions': args.sessions
            }
            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")):
                if args.action in ['tune', 'train']:
                    traindata = MSC_Turn_Facts(subset='train', max_samples=args.train_samples, **dataset_config)
                    validdata = MSC_Turn_Facts(subset='valid', max_samples=args.valid_samples, **dataset_config)
                if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                    testdata = MSC_Turn_Facts(subset='test', max_samples=args.test_samples, **dataset_config)
            collate_fn = MSC_Turn_Facts.batchify
        elif args.task == 'clf_act':
            MSC_SpeechAct.set(tokenizer=tokenizer)
            dataset_config = {
                'basedir': args.datadir + args.basedir,
            }
            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")):
                if args.action in ['tune', 'train']:
                    traindata = MSC_SpeechAct(subset='train', max_samples=args.train_samples, **dataset_config)
                    validdata = MSC_SpeechAct(subset='valid', max_samples=args.valid_samples, **dataset_config)
                if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                    testdata = MSC_SpeechAct(subset='test', max_samples=args.test_samples, **dataset_config)
            collate_fn = partial(MSC_SpeechAct.batchify, with_labels=True, batch_pad_id=tokenizer.pad_token_id)        

    elif args.task in ['generate', 'summarize']: 

        # Generate the fact(s) that are implied by the dialog turns (if any)

        if args.model == "seq2seq":

            if args.load == '':
                tokenizer = train_tokenizer(
                    corpus=MSC_Turns(basedir=args.basedir, session=args.session, subset='train', max_samples=args.train_samples),
                    max_size=args.vocab_size
                )
                if args.add_tokens is not None:
                    num_added_toks = tokenizer.add_tokens(args.add_tokens)
            else:
                tokenizer = Tokenizer.from_pretrained(args.checkpoint_dir + savename(args) + '_tokenizer.json')
            if args.save != '':
                tokenizer.save(args.checkpoint_dir + savename(args) + '_tokenizer')
            pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
            eos_token_id = tokenizer.token_to_id(END_TOKEN)
            unk_token_id = tokenizer.token_to_id(UNK_TOKEN)
            nofact_token_id = tokenizer.token_to_id(args.nofact_token) if args.nofact_token != '' else eos_token_id
            assert nofact_token_id != unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)
            vocab_size = tokenizer.get_vocab_size()
            encoder_opts = {
                "input_size": vocab_size,
                "embedding_size": args.embedding_size,
                "hidden_size": args.hidden_size,
                "aggregate_method": args.aggregate_method
            }
            decoder_opts = {
                "input_size": vocab_size,
                "embedding_size": args.embedding_size,
                "hidden_size": {
                    "mean": args.embedding_size,
                    "lstm": args.hidden_size,
                    "bilstm": args.hidden_size * 2,
                    "poolbilstm": args.hidden_size * 2            
                }[args.encoder],
                "output_size": vocab_size
            }
            model = PersonaExtractor(args.encoder, encoder_opts, args.decoder, decoder_opts, start_token=eos_token_id, nofact_token_id=nofact_token_id)
            criterion = nn.NLLLoss(ignore_index=pad_token_id)

        elif args.model[-4:] == "bart":

            tokenizer = AutoTokenizer.from_pretrained(args.bart_base)
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
            pad_token_id = tokenizer.pad_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
            assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

            if args.model == "bart":
                model = BartExtractor(bart_base=args.bart_base, nofact_token_id=nofact_token_id)
            else:
                model = PrefixBart(
                    bart_base=args.bart_base, 
                    nofact_token_id=nofact_token_id, 
                    freeze=args.freeze, 
                    enc_prefix_size=args.enc_prefix_size,
                    dec_prefix_size=args.dec_prefix_size,
                    prefix_aggr=args.prefix_aggr
                )
            model.bart.resize_token_embeddings(len(tokenizer))
            criterion = ExtractedFactLoss(
                nofact_token_id=nofact_token_id, 
                ignore_index=tokenizer.pad_token_id, 
                lm_weight=args.lm_loss_factor, 
                nofact_weight=args.nofact_weight, 
                num_tokens=len(tokenizer), 
                clf_loss=args.clf_loss
            )

        elif args.model == "t5":

            tokenizer = AutoTokenizer.from_pretrained(args.t5_base, model_max_length=128)
            if args.add_tokens is not None:
                num_added_toks = tokenizer.add_tokens(args.add_tokens)
            pad_token_id = tokenizer.pad_token_id
            nofact_token_id = tokenizer.convert_tokens_to_ids(args.nofact_token) if args.nofact_token != '' else tokenizer.eos_token_id
            assert nofact_token_id != tokenizer.unk_token_id, "nofact_token '{}' must be known token".format(args.nofact_token)

            model = T5Extractor(t5_base=args.t5_base, nofact_token_id=nofact_token_id)
            model.t5.resize_token_embeddings(len(tokenizer))
            criterion = ExtractedFactLoss(
                nofact_token_id=nofact_token_id, 
                ignore_index=tokenizer.pad_token_id, 
                lm_weight=args.lm_loss_factor, 
                nofact_weight=args.nofact_weight, 
                num_tokens=len(tokenizer), 
                clf_loss=args.clf_loss
            )

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

        if args.task == 'generate':
            MSC_Turns.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
            dataset_config = {
                'basedir': args.datadir + args.basedir,
                'sessions': args.sessions
            } 
            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                if args.action in ['tune', 'train']:
                    traindata = MSC_Turns(subset='train', max_samples=args.train_samples, **dataset_config)
                    validdata = MSC_Turns(subset='valid', max_samples=args.valid_samples, **dataset_config)
                if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                    testdata = MSC_Turns(subset='test', max_samples=args.test_samples, **dataset_config)
            collate_fn = partial(MSC_Turns.batchify, with_labels=True, batch_format=model.batch_format, batch_pad_id=pad_token_id)
        else:
            MSC_Summaries.set(tokenizer=tokenizer, len_context=args.len_context, speaker_prefixes=args.speaker_prefixes, nofact_token=args.nofact_token)
            dataset_config = {
                'basedir': args.datadir + args.basedir,
                'session': args.session
            } 
            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                testdata = MSC_Summaries(subset='test', max_samples=args.test_samples, **dataset_config)
            collate_fn = MSC_Summaries.batchify            

    elif args.task == "dialog": # Generate next utterance based on previous dialog turns

        if args.model == "kg_gen":
        
            tokenizer = AutoTokenizer.from_pretrained(args.lm)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if args.add_tokens is not None:
                tokenizer.add_tokens(args.add_tokens)
            tokenizer.bos_token_id = tokenizer.eos_token_id
            model = KnowledgeGroundedDecoder(
                args.lm, tokenizer.bos_token_id, args.fixed_lm, args.num_hops, args.gamma, args.aggregate_method, args.block_src, args.gate,
                tokenizer, 
                config=PretrainedConfig()
            )
            model.gpt2model.resize_token_embeddings(len(tokenizer))
            criterion = KG_loss(ignore_index=tokenizer.pad_token_id, invalid=-1, alpha = args.alpha, beta = args.beta)

            kg = ConceptGraph(args.kg_datadir, args.kg)
            kg.build_reduced_graph(args.kg_datadir + args.dataset_concepts)

            KG_enriched_MSC_Session.set(
                tokenizer, args.speaker_prefixes, 
                kg, args.num_hops, args.max_branch, args.max_concepts, args.max_triples, args.overlapping_concepts
            )

            dataset_config = {
                'basedir': args.datadir + args.basedir,
                'session': args.session if args.session != 1 else '-'.join(['1'] + args.convai2_version),
                'include_persona': args.include_persona
            }

            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                if args.action in ['tune', 'train']:
                    traindata = KG_enriched_MSC_Session(subset='train', max_samples=args.train_samples, **dataset_config)
                    validdata = KG_enriched_MSC_Session(subset='valid', max_samples=args.valid_samples, **dataset_config)
                if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
                    testdata = KG_enriched_MSC_Session(subset='test', max_samples=args.test_samples, **dataset_config)
            collate_fn = partial(KG_enriched_MSC_Session.batchify, batch_format=KnowledgeGroundedDecoder.batch_format)

        elif args.model == "dialogpt":

            tokenizer = AutoTokenizer.from_pretrained(args.lm)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.model_max_length = args.n_positions
            if args.add_tokens is not None:
                tokenizer.add_tokens(args.add_tokens)
            tokenizer.bos_token_id = tokenizer.eos_token_id
            if args.speaker_prefixes is not None:
                tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(args.speaker_prefixes[0])
            model = DialoGPT(args.lm, tokenizer.bos_token_id, args.n_positions)
            model.model.resize_token_embeddings(len(tokenizer))
            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

            MSC_Session.set(
                tokenizer=tokenizer, 
                speaker_prefixes=args.speaker_prefixes, 
                sessionbreak_token=args.sessionbreak_token, 
                speechact_classifier=None if args.speechact_classifier is None else SpeechactClassifier(checkpoint_dir=args.checkpoint_dir, modelname=args.speechact_classifier)
            )
 
            dataset_config = {
                'basedir': args.datadir + args.basedir,
                'session': args.session if args.session != 1 else '-'.join(['1'] + args.convai2_version),
                'include_persona': args.include_persona,
                'include_history': args.include_history,
                'augmented': args.augmented, 
                'selected_turns': args.selected_turns,
                'persona_selector': args.persona_selector,
                'input_order': args.input_order
            }

            if args.include_persona and args.persona_selector is not None:

                # Load pretrained model to select generate (tokens for) persona sentences from a batch with input_ids
                if args.persona_selector == 'init_persona':
                    dataset_config['persona_selector_fn'] = lambda turns: []  # no persona sentences except init_persona
                else:
                    if args.persona_selector.split(':')[0] == 'preprocessed':
                        modelname = args.persona_selector.split(':')[1]
                    else:
                        modelname = args.persona_selector
                    loadpath = args.checkpoint_dir + modelname
                    logging.info("Loading persona_selector from {}".format(loadpath))
                    bart_config = load_config(loadpath + '.config')
                    assert bart_config["speaker_prefixes"] == args.speaker_prefixes, f"persona selector was trained with speaker prefixes {bart_config['speaker_prefixes']}, current dataset has speaker prefixes {args.speaker_prefixes}"
                    bart_tokenizer = AutoTokenizer.from_pretrained(bart_config['bart_base'])
                    if bart_config['add_tokens'] is not None:
                        bart_tokenizer.add_tokens(bart_config['add_tokens'])
                    bart_nofact_token_id = bart_tokenizer.convert_tokens_to_ids(bart_config['nofact_token']) if bart_config['nofact_token'] != '' else bart_tokenizer.eos_token_id
                    bart_model = BartExtractor(bart_config['bart_base'], bart_nofact_token_id)
                    bart_model.bart.resize_token_embeddings(len(bart_tokenizer))
                    bart_generation_config = {
                        "num_beams": bart_config.get('num_beams', 5),
                        "do_sample": bart_config.get('do_sample', True),
                        "temperature": bart_config.get('temperature', 1.0),
                        "top_p": bart_config.get('top_p', 1.0),
                        "top_k": bart_config.get('top_k', 50),
                        "max_new_tokens": bart_config.get('decoder_max', 30),
                    }
                    bart_device = args.device
                    if bart_device == 'mps':
                        bart_device = 'cpu'
                        logging.warning("Changed device from 'mps' to 'cpu' for BART persona selector")
                    bart_model.load_state_dict(torch.load(loadpath, map_location=torch.device(bart_device)))

                    # Configure MSC_Turns to predict persona sentences from a list of utterances
                    MSC_Turns.set(
                        tokenizer=bart_tokenizer, 
                        len_context=2, 
                        speaker_prefixes=bart_config['speaker_prefixes'], 
                        nofact_token=bart_config['nofact_token']
                    )
                    dataset_config['persona_selector_fn'] = partial(
                        MSC_Turns.predict_from_utterances, 
                        model=bart_model, 
                        generation_config=bart_generation_config,
                        device=bart_device, 
                        batch_size=args.batch_size
                        )

            with FileLock(os.path.expanduser(args.datadir[:-1] + ".lock")): 
                if args.action in ['tune', 'train']:
                    traindata = MSC_Session(subset='train', max_samples=args.train_samples, **dataset_config)
                    validdata = MSC_Session(subset='valid', max_samples=args.valid_samples, **dataset_config)
                if args.action in ['eval', 'chat', 'selfchat'] or (args.action =='train' and (not args.skip_eval)):
                    testdata = MSC_Session(subset='test', max_samples=args.test_samples, **dataset_config)
                if args.action == 'selfchat':
                    dataset_config_other = dataset_config
                    if not args.new_agent:
                        dataset_config_other['flipped_perspective'] = True
                    else:
                        dataset_config_other.update({
                            'include_persona': args.include_persona_other,
                            'include_history': args.include_history_other,
                            'input_order': args.input_order_other,
                            'persona_selector': args.persona_selector_other,
                            'persona_selector_fn': None,
                            'flipped_perspective': True
                        })
                        if args.include_persona_other and args.persona_selector_other is not None:
                            if args.persona_selector_other == 'init_persona':
                                dataset_config['persona_selector_fn'] = lambda turns: []  # no persona sentences except init_persona
                            else:
                                logging.warning(f"Persona selection with {args.persona_selector_other} not available for second agent, using gold summaries instead")
                    testdata_other = MSC_Session(subset='test', max_samples=testdata.indices, **dataset_config_other)
                    testdata = (testdata, testdata_other)
            collate_fn = partial(MSC_Session.batchify, with_labels=True, batch_format=DialoGPT.batch_format, batch_pad_id=tokenizer.pad_token_id, buffer=0)

        else:
            assert False, "Model {} is incompatible with task {}".format(args.model, args.task)

    return model, traindata, validdata, testdata, collate_fn, criterion


def train_with_args(config, args):

    if config is not None:
        for k, v in config.items():
            logging.info("Override/set {} to {}".format(k, v))
            setattr(args, k, v)

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Set up for model: {}, task: {}".format(args.model, args.task))
    model, traindata, validdata, testdata, collate_fn, criterion = prepare_model_and_data(args)

    if args.use_wandb:
        wandb.init(project="pex", entity="thegist")
        wandb.config.update(args)  # Converts args to a dictionary and logs it in wandb

    if args.load != "":
        loadpath = args.checkpoint_dir + args.load
        logging.info("Loading model from {}".format(loadpath))
        model.load_state_dict(torch.load(loadpath, map_location=torch.device(args.device)))

    stats = {}
    if args.action in ['tune', 'train']: 
        train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = torch.utils.data.DataLoader(dataset=validdata, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.valid_interval is None:
            args.valid_interval = len(train_loader)

        logging.info("Start training")
        logging.info(f"Use train/valid dataset with {len(traindata)}/{len(validdata)} samples")
 
        model, valid_stats = train(
            model, train_loader, valid_loader, optimizer, criterion, 
            device=args.device, epochs=args.epochs, log_interval=args.log_interval, valid_interval=args.valid_interval, patience=args.patience,
            do_tune=args.action == 'tune', use_wandb=args.use_wandb
        )
        stats = valid_stats

        if args.action == 'train' and args.save != "":
            savepath = args.checkpoint_dir + savename(args)
            logging.info("Saving model to {}".format(savepath))
            torch.save(model.state_dict(), savepath)
            save_config(savepath + '.config', args)

    if args.action in ['train', 'eval'] and not args.skip_eval:

        logging.info("Start testing")
        if args.task != 'summarize':
            logging.info(f"Use test dataset with {len(testdata)} samples")
            test_loader = torch.utils.data.DataLoader(dataset=testdata, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)  
            test_stats = valid(model, test_loader, criterion, device=args.device)

            logging.report("Test stats: {}".format(test_stats))
            stats.update(dict_with_key_prefix(test_stats, prefix="test_"))

            if args.use_wandb:
                wandb.run.summary["test_accuracy"] = stats["test_acc"]  

        eval_stats, result_dict = evaluate(model, testdata, args)

        savepath = args.output_dir + (args.load if args.load != "" else savename(args)) + datetime.now().strftime("_%Y%m%d_%H%M%S") + "_evalresults"
        save_dict(savepath, result_dict, config=vars(args))
        logging.info(f"Saved evalresults in {savepath}")
        stats.update(dict_with_key_prefix(eval_stats, prefix="eval_"))

    if args.action == 'chat':
        assert args.task == 'dialog', f"Chat not compatible with task '{args.task}'; choose 'dialog'"
        logging.info("Start chat")
        chat_stats, result_dict = chat(model, testdata, args)

        savepath = args.output_dir + (args.load if args.load != "" else savename(args)) + datetime.now().strftime("_%Y%m%d_%H%M%S") + "_chatresults"
        save_dict(savepath, result_dict, config=vars(args))
        logging.info(f"Saved chat results in {savepath}")
        stats.update(dict_with_key_prefix(chat_stats, prefix="chat_"))

    if args.action == 'selfchat':
        assert args.task == 'dialog', f"Self chat not compatible with task '{args.task}'; choose 'dialog'"
        logging.info("Start self_chat")

        if args.load_other == "":
            logging.info("Using same model for both agents")
            model_other = model
        else:
            loadpath = args.checkpoint_dir + args.load_other
            logging.info("Loading model for other agent from {}".format(loadpath))
            model_other = copy.deepcopy(model)
            model_other.load_state_dict(torch.load(loadpath, map_location=torch.device(args.device)))
        selfchat_stats, result_dict = selfchat((model, model_other), testdata, args)

        savepath = args.output_dir + (args.load if args.load != "" else savename(args)) + datetime.now().strftime("_%Y%m%d_%H%M%S") + "_selfchatresults"
        save_dict(savepath, result_dict, config=vars(args))
        logging.info(f"Saved selfchat results in {savepath}")
        stats.update(dict_with_key_prefix(selfchat_stats, prefix="selfchat_"))

    return stats


def get_args():

    parser = argparse.ArgumentParser(description="PERSONA EXTRACTOR (note: models and tasks have additional options, please consult the documentation)", conflict_handler="resolve")

    # General, loading, saving, logging
    generalgroup = parser.add_argument_group("general options and setting for loading, saving, monitoring")
    generalgroup.add_argument("--configfile", is_config_file=True, help="configfile with default value (will be overridden by cmdline arguments)")
    generalgroup.add_argument("--seed", type=int, default=42, help="random seed")
    generalgroup.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    generalgroup.add_argument("--output_dir", type=str, default="./output/")
    generalgroup.add_argument("--log_interval", type=int, default=10, help="report interval")
    generalgroup.add_argument("--loglevel", type=str, default="INFO", choices=logging.get_all_levels())    
    generalgroup.add_argument("--logdir", type=str, default=None, help="directory for logfiles; None means no logfile")
    generalgroup.add_argument("--load", type=str, default="", help="filename of model to load")
    generalgroup.add_argument("--save", type=str, default="", help="filename to save the model")
    generalgroup.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    generalgroup.add_argument("--use_wandb", default=False, action='store_true')

    # Main arguments
    parser.add_argument("action", type=str, choices=['tune', 'train', 'eval', 'chat', 'selfchat'], help="choose an action")
    parser.add_argument("model", type=str, choices=["seq2seq", "bert", "bart", "prefixbart", "t5", "kg_gen", "dialogpt"], help="choose one of the available models")
    parser.add_argument("task", type=str, choices=["generate", "summarize", "classify", "clf_act", "dialog"], help="choose a task/dataset to use for tuning/training/evaluation")

    tune_group = parser.add_argument_group("options for tuning")
    tune_group.add_argument("--experiment_name", type=str, default="trainpex", help="experiment name for Ray Tune")
    
    traingroup = parser.add_argument_group("options for training")
    traingroup.add_argument("--epochs", type=int, default=1, help="number of epochs for training")
    traingroup.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    traingroup.add_argument("--valid_interval", type=int, default=None, help="validation interval")
    traingroup.add_argument("--patience", type=int, default=None, help="number of validation intervals without improvement after which training will be terminated")
    traingroup.add_argument("--batch_size", type=int, default=32, help="batch size")
    traingroup.add_argument("--skip_eval", default=False, action='store_true', help="just train")

    evalgroup = parser.add_argument_group("options for evaluation")
    evalgroup.add_argument("--metrics", nargs='*', help="only report listed metrics")
    evalgroup.add_argument("--print_max", type=int, default=20, help="max number of test examples to print")
    evalgroup.add_argument("--temperature", type=float, default=1.0, help="value used to modulate the next token probabilities")
    evalgroup.add_argument("--top_p", type=float, default=1.0, 
        help="if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation"
    )
    evalgroup.add_argument("--top_k", type=int, default=50, help="the number of highest probability vocabulary tokens to keep for top-k-filtering")
    evalgroup.add_argument("--do_sample", default=False, action='store_true', help="whether or not to use sampling ; use greedy decoding otherwise")
    evalgroup.add_argument("--num_beams", type=int, default=1, help="number of beams for beam search; 1 means no beam search")

    chatgroup = parser.add_argument_group("options for chat")
    chatgroup.add_argument("--chatdialog_id", type=int)
    chatgroup.add_argument("--chatturn_id", type=int)
    chatgroup.add_argument("--user_message", type=str, default=None)
    chatgroup.add_argument("--chat_initfile", type=str, default="")

    selfchatgroup = parser.add_argument_group("options for selfchat")
    selfchatgroup.add_argument("--num_turns", type=int, default=8, help="number of turns generated in the selfchat")
    selfchatgroup.add_argument("--new_agent", default=False, action='store_true', help="it True, specify second agent with its own configuration (default is to use a clone)")
    selfchatgroup.add_argument("--load_other", type=str, default="", help="filename of model to load for second agent")
    selfchatgroup.add_argument("--include_persona_other", default=False, action='store_true')
    selfchatgroup.add_argument("--include_history_other", default=False, action='store_true')
    selfchatgroup.add_argument("--input_order_other", default='history-personas-current')
    selfchatgroup.add_argument("--persona_selector_other", type=str, default=None, help="Model to select relevant persona sentences")
    selfchatgroup.add_argument("--temperature_other", type=float, default=1.0, help="value used to modulate the next token probabilities")
    selfchatgroup.add_argument("--top_p_other", type=float, default=1.0, help="top-p parameter for other agent")
    selfchatgroup.add_argument("--top_k_other", type=int, default=50, help="top-k parameter for other agent")
    selfchatgroup.add_argument("--do_sample_other", default=False, action='store_true', help="whether or not to use sampling for other agent")
    selfchatgroup.add_argument("--num_beams_other", type=int, default=1, help="number of beams for beam search for other agent")

    args = parser.parse_known_args()[0]

    # Add cmdline arguments for model
    modelgroup = parser.add_argument_group("Options for the chosen model")
    {
        "seq2seq": PersonaExtractor,
        "bert": PrefixBert,
        "bart": BartExtractor,
        "prefixbart": PrefixBart,
        "t5": T5Extractor,
        "kg_gen": KnowledgeGroundedDecoder,
        "dialogpt": DialoGPT,
    }[args.model].add_cmdline_args(modelgroup)

    if args.model == "seq2seq":
        modelgroup.add_argument("--vocab_size", type=int, default=None, help="Max number of unique token (excluding special tokens)")

    # Add cmdline arguments for Task/Dataset
    parser.add_argument("--datadir", type=str, default="./data/", help="root directory for the dataset files")
    parser.add_argument("--basedir", type=str, default="msc/msc_personasummary/", help="base directory for dataset")
    parser.add_argument("--train_samples", type=int, default=None, help="max number of training samples")
    parser.add_argument("--valid_samples", type=int, default=None, help="max number of test samples")
    parser.add_argument("--test_samples", type=int, default=None, help="max number of test samples")

    if args.task == "classify":
        parser = MSC_Turn_Facts.add_cmdline_args(parser)
    elif args.task == "clf_act":
        parser = MSC_SpeechAct.add_cmdline_args(parser)
    elif args.task in ["generate", "summarize"]:
        if args.action == 'eval' or (args.action =='train' and (not args.skip_eval)):
            parser = TerpMetric.add_cmdline_args(parser)
            parser = NLIMetric.add_cmdline_args(parser)
        if args.task == 'generate':
            parser = MSC_Turns.add_cmdline_args(parser)
        else:
            parser = MSC_Summaries.add_cmdline_args(parser)
            assert args.action == 'eval', f"Action {args.action} not implemented for task 'summarize'"
    elif args.task == "dialog": 
        if args.model == "kg_gen":
            parser = KG_enriched_MSC_Session.add_cmdline_args(parser)
        elif args.model == "dialogpt":
            parser = MSC_Session.add_cmdline_args(parser)
        args = parser.parse_known_args()[0]
        if args.session == 1:
            parser = ConvAI2.add_cmdline_args(parser)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    # Check availability of requested device
    if args.device == "mps":
        assert torch.backends.mps.is_available(), "Device 'mps' not available"
        assert torch.backends.mps.is_built(), "PyTorch installation was not built with MPS activated"
    elif args.device == "cuda":
        assert torch.cuda.is_available(), "Cuda not available"

    # Prepare logging
    logging.set_log_level(args.loglevel)
    if args.logdir is not None:
        logging.add_file_handler(logdir=args.logdir)
    logging.info(prettydict(vars(args), title="Args"))

    if args.action == 'tune':
        ray_dir = args.output_dir + "ray_results"
        trainable = partial(train_with_args, args=args)
        if args.device == 'cuda':
            trainable = with_resources(trainable, {"gpu": 1})
        run_config = RunConfig(
            local_dir=ray_dir,
            name=args.experiment_name 
        )
        results = do_tune(train_fn=trainable, run_config=run_config)
        save_config(f"{ray_dir}/{args.experiment_name}/base.config", args)
        logging.info(f"Ray results saved in {ray_dir}/{args.experiment_name}")
    else:
        stats = train_with_args(config=None, args=args)
        logging.success(prettydict(stats, title="Overview of stats"))

        savepath = args.output_dir + (args.load if args.load != "" else savename(args)) + datetime.now().strftime("_%Y%m%d_%H%M%S") + "_stats"
        save_dict(savepath, stats, config=vars(args))
        logging.info(f"Stats saved in {savepath}")


