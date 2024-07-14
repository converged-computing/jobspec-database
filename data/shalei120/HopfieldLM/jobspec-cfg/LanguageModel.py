import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime
from Hyperparameters import args
from queue import PriorityQueue
import copy

# from kenLM import LMEvaluator as LMEr

from modules import Hopfield, HopfieldPooling, HopfieldLayer
from modules.transformer import  HopfieldEncoderLayer
from HackTransformer import TransformerEncoderLayer,TransformerEncoder
from EnergyTransformer import EnergyTransformerEncoderLayer,EnergyTransformerEncoder
from AMTransformer import AMTransformerEncoderLayer,AMTransformerEncoder

class LanguageModel(nn.Module):
    def __init__(self,w2i, i2w, wordN):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LanguageModel, self).__init__()
        print("LanguageModel creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']
        self.batch_size = args['batchSize']
        self.wordN = nn.Embedding(args['vocabularySize'],1).from_pretrained(torch.FloatTensor(wordN).unsqueeze(1)).to(args['device'])

        self.dtype = 'float32'

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize']).to(args['device'])

        # if args['decunit'] == 'lstm':
        #     self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],
        #                             hidden_size=args['hiddenSize'],
        #                             num_layers=args['dec_numlayer']).to(args['device'])
        # elif args['decunit'] == 'gru':
        #     self.dec_unit = nn.GRU(input_size=args['embeddingSize'],
        #                            hidden_size=args['hiddenSize'],
        #                            num_layers=args['dec_numlayer']).to(args['device'])

        # self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize']).to(args['device'])
        self.logsoftmax = nn.LogSoftmax(dim = -1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        # self.init_state = (torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=args['device']),
        #                    torch.rand(args['dec_numlayer'], 1, args['hiddenSize'], device=args['device']))

        if args['LMtype'] == 'asso':
            self.hopfield = Hopfield(
                input_size=args['embeddingSize'] )
            output_projection = nn.Linear(in_features=self.hopfield.output_size, out_features=args['vocabularySize'])
            self.hp_network = nn.Sequential(self.hopfield, output_projection).to(args['device'])
        elif args['LMtype'] == 'asso_enco':
            self.hopfield = Hopfield(
                input_size=args['embeddingSize'] )
            self.hp_network1 = HopfieldEncoderLayer(self.hopfield)
            self.hp_network =  nn.TransformerEncoder(self.hp_network1, num_layers=args['numLayers']).to(args['device'])
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize'])
        elif args['LMtype'] == 'bigbird':
            self.trans_net = nn.TransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(
                args['device'])
            self.transformer_encoder = nn.TransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(
                args['device'])
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize'])
        elif args['LMtype'] == 'transformer':
            # self.trans_net = nn.TransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(args['device'])
            # self.transformer_encoder = nn.TransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(args['device'])
            self.trans_net =  TransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(args['device'])
            self.transformer_encoder =  TransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(args['device'])
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize'])
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(args['device'])
        elif args['LMtype'] == 'energy':
            self.trans_net = EnergyTransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(args['device'])
            self.energytransformer_encoder = EnergyTransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(args['device'])
            self.output_projection = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize']).to(args['device'])
            # self.transformer_network = nn.Sequential(self.transformer_encoder, output_projection).to(args['device'])
            # self.energytransformer_encoder_neg = EnergyTransformerEncoder(self.trans_net, num_layers=args['numLayers'], choice = 0).to(args['device'])
            # self.output_projection_neg = nn.Linear(in_features=args['embeddingSize'], out_features=args['vocabularySize']).to(args['device']

        elif args['LMtype'] == 'hop_energy':
            self.trans_net = AMTransformerEncoderLayer(d_model=args['embeddingSize'], nhead=args['nhead']).to(
                args['device'])
            self.AMtransformer_encoder = AMTransformerEncoder(self.trans_net, num_layers=args['numLayers']).to(
                args['device'])
            self.output_projection = nn.Linear(in_features=args['embeddingSize'],
                                               out_features=args['vocabularySize']).to(args['device'])


    def generate_square_subsequent_mask(self, sz):
        # mask = torch.logical_not(torch.triu(torch.ones(sz, sz)) == 1)
        # mask[0, 0] = True
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_bigbird_mask(self, sz):
        # mask = torch.logical_not(torch.triu(torch.ones(sz, sz)) == 1)
        # mask[0, 0] = True
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask2 = torch.zeros(sz, sz)
        mask2[4:,2:] = mask[:-4,:-2]
        x = [np.random.randint(sz) for _ in range(30)]
        y = [np.random.randint(sz) for _ in range(30)]
        mask2[x,y] = 0
        mask = (mask==1) & (mask2 ==0)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def build(self, x, training):
        # print(args['device'])
        self.decoderInputs = x['dec_input'].to(args['device'])
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target'].to(args['device'])

        wordcounts = self.wordN(self.decoderInputs)
        # print(wordcounts)

        batch_size = self.decoderInputs.size()[0]
        self.dec_len = self.decoderInputs.size()[1]
        self.dec_input_embed = dec_input_embed = self.embedding(self.decoderInputs)
        mask = torch.sign(self.decoderTargets.float())

        if args['LMtype']== 'lstm':
            init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))
            de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, self.dec_len)
        elif args['LMtype']== 'asso':
            # print(args['maxLengthDeco'], dec_input_embed.size())
            de_outputs = self.hp_network(dec_input_embed)
        elif args['LMtype']== 'asso_enco':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(args['device'])
            # print(args['maxLengthDeco'], dec_input_embed.size(), src_mask.size())
            de_outputs = self.hp_network(dec_input_embed,src_mask)
            de_outputs = self.output_projection(de_outputs)
            # de_outputs = de_outputs.transpose(0,1)
        elif args['LMtype'] == 'bigbird':
            src_mask = self.generate_bigbird_mask(self.dec_len).to(args['device'])
            de_outputs = self.transformer_encoder(dec_input_embed.transpose(0, 1), mask=src_mask)
            de_outputs = self.output_projection(de_outputs)
            de_outputs = de_outputs.transpose(0, 1)
        elif args['LMtype']== 'transformer':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(args['device'])
            de_outputs, attns= self.transformer_encoder(dec_input_embed.transpose(0,1), mask=src_mask)
            de_outputs = self.output_projection(de_outputs)
            de_outputs = de_outputs.transpose(0,1)
            sum_amloss = 0
            # print(de_outputs.size())
        elif args['LMtype'] == 'energy':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(args['device'])
            _, loss_tuple, error, de_outputs_list, attns  = self.energytransformer_encoder(dec_input_embed, mask = src_mask, src_key_padding_mask = mask, training=training)
            de_outputs = self.output_projection(de_outputs_list[-1])
            # de_outputs_neg, loss_tuple_neg, error_neg = self.energytransformer_encoder_neg(dec_input_embed, mask = src_mask, src_key_padding_mask = mask, training=training)
            # de_outputs_neg = self.output_projection(de_outputs_neg)
            # de_outputs = de_outputs.transpose(0,1)

        elif args['LMtype'] == 'hop_energy':
            src_mask = self.generate_square_subsequent_mask(self.dec_len).to(args['device'])
            de_outputs, sum_amloss = self.AMtransformer_encoder(dec_input_embed.transpose(0,1),
                                      mask=src_mask, src_key_padding_mask=(1-mask).to(torch.bool), wordN = wordcounts)
            de_outputs = self.output_projection(de_outputs)
            de_outputs = de_outputs.transpose(0,1)
        # print(de_outputs.size(),self.decoderTargets.size())
        # print(de_outputs.size(),self.decoderTargets.size() )
        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        recon_loss = torch.squeeze(recon_loss) * mask
        recon_loss_mean = torch.mean(recon_loss, dim=-1)


        # recon_loss_neg = self.CEloss(torch.transpose(de_outputs_neg, 1, 2), self.decoderTargets)
        # recon_loss_neg = torch.squeeze(recon_loss_neg) * mask
        # recon_loss_mean_neg = torch.mean(recon_loss_neg, dim=-1)
        # print(recon_loss.size(), mask.size())
        true_mean = recon_loss.sum(1) / mask.sum(1)

        data = {'de_outputs': de_outputs,
                'amloss': sum_amloss,
                'loss':recon_loss_mean + 0.01* sum_amloss,
                'true_mean':true_mean}

        if args['LMtype'] == 'energy':
            targetvector = self.embedding(self.decoderTargets)
            data['KL'] = loss_tuple[1]
            data['VAE_recon']  = loss_tuple[0]
            # recon_loss_mean = recon_loss_mean.mean() + 100*KL
            # c_loss = 0
            # for out in de_outputs_list[:-1]:
            #     dots = torch.einsum('bse,bse->bs',out, targetvector)
            #     # dots1 = torch.exp(-dots.clone())
            #     dots *= mask
            #     c_loss += torch.exp(-dots.mean())

            data['error'] = error
            data['attns'] = attns

        elif args['LMtype'] == 'transformer':
            data['attns'] = attns



        return data

    def forward(self, x):
        data = self.build(x, training=True)
        return data

    def predict(self, x):
        data = self.build(x, training=False)
        return data




    def generate(self, init_state, senlen, assistseq = [], startseq = []):
        self.batch_size = 1
        init_state = (init_state[0].repeat([1, self.batch_size, 1]), init_state[1].repeat([1, self.batch_size, 1]))

        de_words = self.decoder_g(init_state, senlen, assistseq, startseq)

        return de_words

    def decoder_t(self, initial_state, inputs, batch_size, dec_len):
        inputs = torch.transpose(inputs, 0,1).contiguous()
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)
        # output = output.cpu()

        output = self.out_unit(output.view(batch_size * dec_len, args['hiddenSize']))
        output = output.view(dec_len, batch_size, args['vocabularySize'])
        output = torch.transpose(output, 0,1)
        return output, out_state



if __name__ == '__main__':
    LMer = LMEr()
    textData = TextData('LMbenchmark')
    # args['vocabularySize'] = textData.getVocabularySize()
    # model_path = args['rootDir'] + '/model.pth'
    # model = torch.load(model_path, map_location=args['device'])
    # sentoken =  torch.LongTensor([[textData.word2index[w] for w in 'START_TOKEN three of hit by the companies unite the a the next END_TOKEN'.split()]])
    # y_hat_emb = model.embedding(sentoken.to(args['device']))
    # loss = model.Cal_LMloss_with_fuzzyword(y_hat_emb, torch.LongTensor(sentoken).to(args['device']))
    # loss2 = model.Cal_sen_LMloss(y_hat_emb, torch.LongTensor(sentoken).to(args['device']))
    # print(torch.exp(loss), torch.exp(loss2), loss)
    # print(LMer.Perplexity(['three of hit by the companies unite the a the next']))



