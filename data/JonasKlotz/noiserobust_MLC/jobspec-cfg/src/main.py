# for cluster
import sys
sys.path.append("/home/users/j/jonasklotz/remotesensing")

import argparse, warnings
import torch.nn.functional as F

import utils.utils as utils
from utils.collect_env import main as print_env  # for cluster
import torch, torch.nn as nn

from data_pipeline.lmdb_dataloader import load_data_from_lmdb
from data_pipeline.other_dataloaders import load_data_from_dir
from lamp.Models import LAMP, ResnetBaseLine
from config_args import config_args, get_args
from training import run_model
import numpy as np
from copy import deepcopy
from predict import predict
from datetime import datetime
from losses import AsymmetricLoss
from wordembedding.glove import load_word_embeddings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)

torch.manual_seed(42)


def main(opt):
    """ Main function that starts the training and handles everything """

    # printing device info
    # print_env()

    # ========= Loading Dataset =========#
    opt.max_token_seq_len_d = opt.max_ar_length
    print(f"load {opt.dataset}")
    if opt.dataset == "data/apparel":
        train_data, valid_data, test_data, labels = load_data_from_dir(
            data_dir='data/apparel-images-dataset', batch_size=opt.batch_ize)
    else:
        train_data, valid_data, test_data, labels = load_data_from_lmdb(
            data_dir=opt.dataset_path, batch_size=opt.batch_size, add_noise=opt.add_noise, sub_noise=opt.sub_noise)
    n_output_classes = len(labels)

    opt.tgt_vocab_size = n_output_classes  # number of labels
    label_adj_matrix = torch.ones(opt.tgt_vocab_size, opt.tgt_vocab_size)  # full graph

    # ========= Preparing Model =========#
    print(f"Using Model: {opt.model}")
    if opt.model == "resnet_base":
        model = ResnetBaseLine(d_model=n_output_classes, resnet_layers=18)
    else:
        if opt.model == "lamp":
            # load node embeddings from gauss distribution
            opt.word_embedding_matrix = torch.FloatTensor(np.random.normal(scale=0.6, size=(opt.tgt_vocab_size, opt.d_model)))
        else:
            # load node embeddings from glove
            try:
                opt.word_embedding_matrix = torch.from_numpy(
                    load_word_embeddings(data_path=opt.embedded_weights_path, dim=opt.d_model, labels=labels)) \
                    .to(torch.float32)

            except FileNotFoundError as e:
                print(f"ERROR: Glovefile not found {e}\n"
                      f"defaulting to normal distributed weights")
                opt.word_embedding_matrix = torch.FloatTensor(np.random.normal(scale=0.6, size=(opt.tgt_vocab_size, opt.d_model)))
        # copy embedings for regularization
        opt.original_word_embedding_matrix = deepcopy(opt.word_embedding_matrix)

        model = LAMP(opt.tgt_vocab_size, opt.max_token_seq_len_d, n_layers_dec=opt.n_layers_dec, n_head=opt.n_head,
                     n_head2=opt.n_head2, d_word_vec=opt.d_model, d_model=opt.d_model, d_inner_hid=opt.d_inner_hid,
                     d_k=opt.d_k, d_v=opt.d_v, dec_dropout=opt.dec_dropout, dec_dropout2=opt.dec_dropout2,
                     proj_share_weight=opt.proj_share_weight,
                     enc_transform=opt.enc_transform, onehot=opt.onehot, no_dec_self_att=opt.no_dec_self_att,
                     loss=opt.loss,
                     label_adj_matrix=label_adj_matrix, label_mask=opt.label_mask, graph_conv=opt.graph_conv,
                     attn_type=opt.attn_type, int_preds=opt.int_preds, word2vec_weights=opt.word_embedding_matrix)



    opt.total_num_parameters = int(utils.count_parameters(model))

    if opt.load_emb:
        model = utils.load_embeddings(model, '../../Data/word_embedding_dict.pth')


    ################## SETUP OPTIMIZER ##############################
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.get_trainable_parameters(), betas=(0.9, 0.999), lr=opt.lr, weight_decay=1e-5)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
                                    momentum=opt.momentum)
    scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,
                                                      last_epoch=-1)

    ################## SETUP LOSS ##############################
    if opt.loss == 'asl':
        print('using ASL')
        crit = AsymmetricLoss(gamma_neg=opt.asl_ng, gamma_pos=opt.asl_pg, clip=opt.asl_clip, eps=opt.asl_eps)
    elif opt.loss == 'weighted_bce':
        print('using weighted BCE ')
        # weight each loss
        # pos_weight_old = torch.tensor([5.8611238, 1.21062702, 5.82371649, 9.89122553,
        #                            14.41991786, 9.75859599])  # for random deepglobe sampling

        pos_weight = torch.tensor([1.9643, 1.1112, 2.0769, 3.4269, 6.2885, 3.6217])

        crit = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    else:
        print("Using BCE")
        crit = nn.BCEWithLogitsLoss(reduction='mean')

    ################## manage CUDA ################

    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if torch.cuda.is_available() and opt.cuda:
        model = model.cuda()
        crit = crit.cuda()
        if opt.gpu_id != -1:
            torch.cuda.set_device(opt.gpu_id)

    ######################## Load a model  ##########################
    if opt.predict:
        predict(model, valid_data, labels, crit, weights_path="results/" +
                                                              "deepglobe/deepglobe.glove_d_300.epochs_5.loss_ce.adam.lr_0001" +
                                                              "/model.chkpt", n=5)
        exit()

    try:
        print("============== Start Training ======================")
        start_time = datetime.now()
        run_model(model=model, train_data=train_data, test_data=test_data, valid_data=valid_data, crit=crit,
                  optimizer=optimizer, scheduler=scheduler, opt=opt, class_names=labels)
        end_time = datetime.now()
        print(f"Total time taken: {end_time - start_time}")

    except KeyboardInterrupt:
        print('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    main(opt)
