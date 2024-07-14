# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha
import functools
print = functools.partial(print, flush=True)
from LanguageModel import LanguageModel
from textdata_wiki2 import  TextData_wiki2
from textdata import  TextData_1mb
from textdata_enwiki8 import  TextData_enwiki8
import time, sys,datetime
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import  tqdm
import os
import math,random
import nltk
from nltk.corpus import stopwords
import argparse
# import GPUtil
# import turibolt as bolt
import pickle
from Hyperparameters import args
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# import seaborn
# seaborn.set_context(context="talk")
# import matplotlib.pyplot as plt
import numpy as np
import copy

# from kenLM import LMEvaluator as LMEr

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--batch', '-b')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--data', '-d')
parser.add_argument('--server', '-s')
parser.add_argument('--embeddingsize', '-emb')
parser.add_argument('--layernum', '-layer')
parser.add_argument('--nhead', '-nhead')
parser.add_argument('--choose', '-c')

cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
    args['device'] = 'cpu'
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.batch is None:
    pass
else:
    args['batchSize'] = int(cmdargs.batch)

if cmdargs.modelarch is None:
    pass
else:
    args['LMtype'] = cmdargs.modelarch

if cmdargs.data is None:
    pass
else:
    args['corpus'] = cmdargs.data

if cmdargs.server is None:
    args['server'] = 'other'
else:
    args['server'] = cmdargs.server

if cmdargs.embeddingsize is None:
    pass
else:
    args['embeddingSize'] = int(cmdargs.embeddingsize)


if cmdargs.layernum is None:
    pass
else:
    args['numLayers'] = int(cmdargs.layernum)
if cmdargs.nhead is None:
    args['nhead'] = 1
else:
    args['nhead'] = int(cmdargs.nhead)
if cmdargs.choose is None:
    args['choose'] = None
else:
    args['choose'] = cmdargs.choose

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), datetime.datetime.now())

class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/model.pth'
        # self.model_bw_path = args['rootDir'] + '/model_bw.pth'
        self.drawcount = 0



    def main(self):
        if args['corpus'] == '1mb':
            self.textData =  TextData_1mb('LMbenchmark')
            # args['embeddingSize'] = 300
        elif  args['corpus'] == 'wiki2':
            self.textData = TextData_wiki2('wiki2')
        elif  args['corpus'] == 'wiki103':
            self.textData = TextData_wiki2('wiki103')
        elif  args['corpus'] == 'enwiki8':
            self.textData = TextData_enwiki8('enwiki8')
        # self.LMer = LMEr()
        self.start_token = self.textData.word2index['START_TOKEN']
        self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        print(self.textData.getVocabularySize())
        print(args)
        torch.manual_seed(0)
        self.model = LanguageModel(self.textData.word2index, self.textData.index2word, wordN= self.textData.word2count).to(args['device'])
        # self.model = torch.load(args['rootDir']+ 'model_energy_wiki2_100_6_300.pth', map_location=args['device'])
        # self.model = torch.load(args['rootDir']+ 'model_transformer_wiki2_100_6_300.pth', map_location=args['device'])
        params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print(params, sum([sys.getsizeof(p.storage()) for p in self.model.parameters()])/1000000, 'm')
        self.trainLM()     # contain  model saving
        self.evaluateRandomly()
        self.testLM()

     

    def trainLM(self,   print_every=10000, plot_every=10, learning_rate=0.001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        print(type(self.textData.word2index), args['device'])

        optimizer = optim.Adam(self.model.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=5.0)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        print('niters ',n_iters)

        args['trainseq2seq'] = False

        min_perplexity = -1

        self.Cal_perplexity_for_dataset('test', False)

        CE_loss_total = 0
        KL_total = 0
        VAE_recon_total = 0
        error_total = 0
        for epoch in range(args['numEpochs']):
            losses = []

            for batch in batches:
                optimizer.zero_grad()
                x={}
                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs))
                x['dec_len'] = batch.decoder_lens
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs))


                # print(x['enc_input'][0],x['dec_input'][0],x['dec_target'][0])
                if args['LMtype'] == 'energy':
                    data = self.model(x)    # batch seq_len outsize
                    loss_mean = torch.mean(data['loss']) + 0.1 * data['KL'] + 0.01 * data['VAE_recon'] + 1 * data['error']
                    CE_loss_total += torch.mean(data['loss']).item()
                    KL_total += data['KL'].item()
                    VAE_recon_total += data['VAE_recon'].item()
                    error_total += data['error'].item()
                else:
                    data = self.model(x)    # batch seq_len outsize
                    loss_mean = torch.mean(data['loss'])
                # Reward = loss_mean.data

                loss_mean.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

                optimizer.step()


                print_loss_total += loss_mean.item()
                plot_loss_total += loss_mean.item()

                losses.append(loss_mean.item())

                # GPUtil.showUtilization()
                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    KL_avg = KL_total / print_every
                    KL_total = 0
                    CEloss_avg = CE_loss_total / print_every
                    CE_loss_total = 0
                    VAE_recon_avg = VAE_recon_total / print_every
                    VAE_recon_total = 0
                    error_avg = error_total / print_every
                    error_total = 0
                    print('%s (%d %d%%) loss = %.4f, CE_loss = %.4f, VAE recon = %.4f, amloss = %.4f, error=%.4f' % (timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg, CEloss_avg, VAE_recon_avg, data['amloss'], error_avg))
                    # GPUtil.showUtilization()
                    # del de_output,loss,true_mean
                    # GPUtil.showUtilization()
                    # torch.cuda.empty_cache()
                    # GPUtil.showUtilization()
                    if args['corpus'] != '1mb' or iter % 100000 == 0:
                        perplexity= self.Cal_perplexity_for_dataset('test', False)
                        print('Test ppl: ', perplexity)
                        # print('Attention prop: ', attn)
                        # print('Attention other: ', other_attn)
                        # print('std other: ', std)
                        # print('KLC kln: ', klc,kln)

                        if perplexity < min_perplexity or min_perplexity == -1:
                            print('perplexity = ', perplexity, '>= min_perplexity (', min_perplexity, '), saving model...')
                            min_perplexity = perplexity

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                iter+=1

            # scheduler.step()
            perplexity= self.Cal_perplexity_for_dataset('test', False)
            if perplexity < min_perplexity or min_perplexity == -1:
                print('perplexity = ', perplexity, '>= min_perplexity (', min_perplexity, '), saving model...')
                torch.save(self.model, self.model_path.replace('model', 'model_'+args['LMtype']+'_'+args['corpus'] + '_'+str(args['maxLength'])+'_'+str(args['numLayers'])+'_'+str(args['embeddingSize'])))
                min_perplexity = perplexity


            # if args['LMtype'] == 'energy':
            #     print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid perplexity = ', perplexity, 'VAE recon = ', VAE_recon_avg, 'KL loss = ', KL_avg)
            # else:
            print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid perplexity = ', perplexity)#, 'Attention prop: ', attn, 'Attention other: ', other_attn,'std other: ', std, 'KLC kln: ', klc,kln)


        # self.test()
        # showPlot(plot_losses)

    def testLM(self):
        start = time.time()
        print('Test set BLEU = ', self.Cal_perplexity_for_dataset('test', printlog = True))
        end = time.time()
        print('Test time: ', time.strftime("%H:%M:%S", time.gmtime(end-start )))

    def Cal_perplexity_for_dataset(self, datasetname, printlog = False):
        if not hasattr(self, 'testbatches' ):
            self.testbatches = {}
        if datasetname not in self.testbatches:
            self.testbatches[datasetname] = self.textData.getBatches(datasetname)
        self.model.eval()
        num = 0
        ave_loss = 0
        if printlog:
            filepointer = open(args['rootDir'] + 'tf_attn_log.txt', 'w')
        with torch.no_grad():
            # print(len(self.testbatches[datasetname][0].decoderSeqs))
            attns = []
            other_attns = []
            stds = []
            klcs = []
            klns = []
            for batch in self.testbatches[datasetname]:
                x = {}

                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs))
                x['dec_len'] = batch.decoder_lens
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs))

                # print('here')
                # GPUtil.showUtilization()
                data = self.model.predict(x)    # batch seq_len outsize
                # GPUtil.showUtilization()
                # true_mean = recon_loss_mean
                # print(true_mean.size())
                sum_true = data['true_mean'].sum().item()
                ave_loss = (ave_loss * num + sum_true) / (num + len(data['true_mean']))

                num += len(data['true_mean'])

                # attn, other_attn, std, klc,kln = self.analysis_attn(data['attn_list'], x['dec_input'], self.model.embedding)
                # attns.append(attn)
                # other_attns.append(other_attn)
                # stds.append(std)
                # klcs.append(klc)
                # klns.append(kln)
                if printlog:
                    self.printattn(data['attns'], batch.raw, filepointer)
        self.model.train()
        return np.exp(ave_loss)#, sum(attns) / len(attns), sum(other_attns) / len(other_attns), sum(stds) / len(stds), sum(klcs) / len(klcs), sum(klns) / len(klns)


    def indexesFromSentence(self,  sentence):
        return [self.textData.word2index[word] if word in self.textData.word2index else self.textData.word2index['UNK'] for word in sentence]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        # indexes.append(self.textData.word2index['END_TOKEN'])
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def evaluate(self,  sentence, correctlabel, max_length=20):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence( sentence)
            input_length = input_tensor.size()[0]
            # encoder_hidden = encoder.initHidden()

            # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            x={}
            # print(input_tensor)
            x['enc_input'] = torch.transpose(input_tensor, 0,1)
            x['enc_len'] = [input_length]
            x['labels'] = [correctlabel]
            # print(x['enc_input'], x['enc_len'])
            # print(x['enc_input'].shape)
            decoded_words, label,_ = self.model.predict(x, True)

            return decoded_words, label

    def evaluateRandomly(self, n=10):
        for i in range(n):
            sample = random.choice(self.textData.datasets['train'])
            print('>', sample)
            output_words, label = self.evaluate(sample[2], sample[1])
            output_sentence = ' '.join(output_words[0])  # batch=1
            print('<', output_sentence, label)
            print('')

    def draw_attn(self, attn, sentences, id=1):
        import matplotlib.pyplot as plt
        from matplotlib import colors

        fontsize = 5
        pwargs = {'interpolation': 'nearest'}
        sen = sentences[id]
        sen = list(sen)
        # print(sen)
        # sen = sen[:10]
        num = len(sen)
        sen = [self.textData.index2word[ii] for ii in sen]
        plt.imshow(attn[2].cpu()[id,:num,:num], cmap=plt.get_cmap('Reds'), interpolation='nearest')  # cmap=plt.cm.jet,**pwargs)
        plt.locator_params(axis='y', nbins=num)
        plt.locator_params(axis='x', nbins=num)
        plt.gca().set_xticklabels(tuple(sen), rotation=-15, ha='left',
                                  minor=False, fontsize=fontsize)
        plt.gca().set_yticklabels(tuple(sen), rotation=0, ha='right',
                                  minor=False, fontsize=fontsize)

        plt.savefig(args['rootDir'] + 'attn.png')


    def analysis_attn(self, attn_list, sentences, embedding):
        probs_list = []
        seq_len = attn_list[0].size()[1]
        ones = torch.diag(torch.ones(seq_len)).unsqueeze(0).to(args['device'])

        utriu = 1- torch.triu(torch.ones(seq_len,seq_len))
        utriu = utriu.unsqueeze(0).to(args['device'])
        other_list = []
        std_list = []

        bigtriu = torch.triu(torch.ones(seq_len,seq_len)).transpose(0,1).unsqueeze(0).to(args['device'])
        sen_embed = embedding(sentences.to(args['device'])) # bse
        sa = torch.einsum('bse,bte->bst', sen_embed, sen_embed)

        current_comp = sa + bigtriu.float().masked_fill(bigtriu == 0, float('-inf')).masked_fill(bigtriu == 1, float(0.0))
        next_comp = torch.clone(sa)
        next_comp[:,:-1,:] = next_comp[:,1:,:]
        next_comp = next_comp + bigtriu.float().masked_fill(bigtriu == 0, float('-inf')).masked_fill(bigtriu == 1, float(0.0))
        # print(current_comp)
        current_comp = F.softmax(current_comp, dim = -1)
        next_comp = F.softmax(next_comp, dim = -1)
        # print(current_comp)

        klcs=[]
        klns = []

        for attn in attn_list:
            probs = attn * ones
            probs = probs.sum(2).mean()
            probs_list.append(probs)

            other_ori = attn * utriu
            tri_num = (seq_len**2 -seq_len) / 2
            tri_mean = other_ori.sum(2).sum(1) / tri_num
            other = tri_mean.mean()
            other_list.append(other)

            mean = other_ori.sum(2)[:,1:] / utriu.sum(2)[:,1:]
            res1 = ((attn[:,1:,:] - mean.unsqueeze(2))**2) * utriu[:,1:,:]
            res1 = res1.sum(2) / utriu.sum(2)[:,1:]
            # print(res1)
            std = torch.sqrt(res1).mean()
            std_list.append(std)

            # for a in attn[5,:,:]:
            #     for b in a:
            #         print(b.cpu().numpy(), end=',')
            #     print()
            # print(attn[5,:,:])
            # print(attn,current_comp)
            klc = attn * torch.log(attn / (current_comp+1e-6) + 1e-6)
            # print('klc',klc)
            klc = klc.sum(2).mean()

            kln = attn * torch.log(attn / (next_comp+1e-6) + 1e-6)
            kln = kln.sum(2).mean()

            klcs.append(klc)
            klns.append(kln)
        return torch.stack(probs_list), torch.stack(other_list), torch.stack(std_list), torch.stack(klcs), torch.stack(klns)

    def printattn(self, attn_list, raw_sentence_list, fp):
        res   = []
        for ind, sen in enumerate(raw_sentence_list):
            res.append([])
            sen = ['SOS'] + sen
            sen = sen[:50]
            for layer_index, attn_layer in enumerate(attn_list):

                res[ind].append({'sen': sen, 'pos':[]})
                matrix = attn_layer[ind,:,:]
                for p in range(10, len(sen)):
                    attn_sen = matrix[p,:]
                    _, indices = torch.topk(attn_sen, 5)
                    high = [sen[i] for i in indices]
                    res[ind][layer_index]['pos'].append((p,sen[p],','.join(high)))


                    fp.write(str(layer_index) + '\t' + ' '.join(sen) + '\t')
                    fp.write(str(p) + ' ' + sen[p] + '\t' + ','.join(high)+'\n')











 

if __name__ == '__main__':
    # args['corpus'] = 'wiki2'
    # args['LMtype'] = 'energy'
    # args['choose'] = 'BET-NN'
    # args['norm_attn'] = True
    r = Runner()
    # r.textData = TextData('LMbenchmark')

    # r.Get_KeyWords(3000)
    r.main()