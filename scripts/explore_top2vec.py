#!/usr/bin/env python3


import argparse
import os
import sys
from top2vec import Top2Vec

here = os.path.abspath(os.path.dirname(__file__))


def get_parser():
    parser = argparse.ArgumentParser(description="top2vec")
    parser.add_argument(
        "--model",
        help="model input file",
        default=os.path.join(here, "data", "combined", "wordclouds", "top2vec.model"),
    )
    parser.add_argument(
        "--outname",
        help="basename of output markdown",
        default="top2vec-jobspec-database.md",
    )
    return parser


def main():
    """
    Generate cosine distance matrix from vectors.
    """
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Note that I had to run this on a large VM to not get killed
    model_path = args.model
    if not os.path.exists(model_path):
        sys.exit(f"Model path {model_path} does not exist.")
    model = Top2Vec.load(model_path)

    # Find associated words with topics we choose!
    terms = [
        "io",
        "storage" "network",
        "protocol",
        "communication",
        "power",
        "resource",
        "lack",
        "eager",
        "algorithm",
        "ml",
        "nvidia",
        "SBATCH",
        "PBATCH",
        "COBALT",
        "PBS",
        "OAR",
        "BSUB",
        "FLUX",
        "gpu",
        "gcc",
        "array",
        "module",
        "mpi",
        "roc",
        "cuda",
        "intel",
        "gmx",
        "namd",
        "lammps",
        "laghos",
        "python",
        "amd",
        "amg",
        "cmake",
        "meson",
        "paraview",
        "hpctoolkit",
        "ninja",
        "rust",
        "spack",
        "easybuild",
        "resnet",
        "pytorch",
        "nvidia",
        "tensorflow",
        "pmix",
        "kripke",
        "quicksilver",
        "maestro",
        "snakemake",
        "nextflow",
        "dask",
        "kube",
        "docker",
        "singularity",
        "valgrind",
        "julia",
        "matlab",
        "nccl",
        "ray",
        "fio",
        "ior",
        "petsc",
    ]

    # returns words, word scores
    outdir = os.path.dirname(args.model)
    outfile = os.path.join(outdir, args.outname)
    not_learned = []
    with open(outfile, "w") as fd:
        fd.write("# GitHub JobSpec Similar Terms\n")
        fd.write(
            "\nThese are top 20 similar terms for a select set, not filtering by score."
        )
        for term in terms:
            try:
                words, word_scores = model.similar_words(
                    keywords=[term], keywords_neg=[], num_words=20
                )
            except:
                print(f"Term {term} was not learned by the model")
                not_learned.append(term)
                continue

            print(f"== {term}")
            fd.write(f"\n## {term}")
            fd.write("\n```\n")
            for i, word in enumerate(words):
                print(f"  {word}: {word_scores[i]}")
                fd.write(f"  {word}: {word_scores[i]}\n")
            fd.write("\n```\n")

        fd.write("\n## Not Learned\n")
        fd.write("\n```\n")
        fd.write(" ".join(not_learned))
        fd.write("\n```\n")


if __name__ == "__main__":
    main()
