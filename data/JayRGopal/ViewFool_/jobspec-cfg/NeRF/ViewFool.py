from rendering_image import render_image
import numpy as np
from PIL import Image
#from NES_just_test import NES_search
from NES import NES_search

from datasets.opts import get_opts

args = get_opts()

print()
print("**** params ****")
for k in vars(args):
    print(k,vars(args)[k])
print("****************")
print()

if args.optim_method == 'NES':
    NES_search()
