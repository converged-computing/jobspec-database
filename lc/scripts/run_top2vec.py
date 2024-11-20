#!/usr/bin/env python3


import argparse
import os
import sys
import rse.utils.file as utils
from top2vec import Top2Vec

here = os.path.abspath(os.path.dirname(__file__))


def get_parser():
    parser = argparse.ArgumentParser(description="doc2vec")
    parser.add_argument(
        "--input",
        help="tokenized jobspec files",
        default=os.path.join(here, "jobspec-corpus.txt"),
    )
    parser.add_argument(
        "--speed",
        help="speed for learning mode (learn, deep-learn, fast-learn)",
        default="learn",
    )
    return parser


def main():
    """
    Generate cosine distance matrix from vectors.
    """
    parser = get_parser()
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.input):
        sys.exit(f"{args.input} does not exist.")

    corpus = [x.strip() for x in utils.read_file(args.input)]

    outdir = os.path.dirname(args.input)
    outdir = os.path.join(outdir, "top2vec")

    # Note I was running this on a VM - YOLO
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Note that I had to run this on a large VM to not get killed
    model_path = os.path.join(outdir, f"top2vec-with-doc2vec-{args.speed}.model")
    if not os.path.exists(model_path):
        # Note I ran this on an instance with 32 cores
        model = Top2Vec(
            corpus, speed=args.speed, workers=192, embedding_model="doc2vec"
        )
        model.save(model_path)
    else:
        model = Top2Vec.load(model_path)


if __name__ == "__main__":
    main()
