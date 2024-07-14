#!/usr/bin/env python3
from yolov5 import train
import os
import sys
import argparse


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--resume', action='store_true', help='resume the training')

    args = parser.parse_args(arguments)

    if args.resume:
        train.run(resume = True)
    else:    
        folder_path = os.path.dirname(__file__)
        train.run(imgsz=640, batch=4, epochs = 300, data= os.path.join(folder_path, 'bdd100k.yaml') , weights='yolov5s.pt')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

