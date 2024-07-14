"""
Extended Maximum intensity Projection (EMIP)

Author: Nazanin Moradinasab
"""



import os, json
import torch
import numpy as np
import random

from options import Options
from prepare_data import main as prepare_data



def main():
    opt = Options(isTrain=True)
    opt.parse()

    if opt.train['random_seed'] >= 0:
        print('=> Using random seed {:d}'.format(opt.train['random_seed']))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
        random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])


    # ----- prepare training data ----- #
    print('=> Preparing training samples')
    prepare_data(opt)



if __name__ == '__main__':
    main()
