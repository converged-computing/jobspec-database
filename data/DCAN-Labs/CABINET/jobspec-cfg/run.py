#!/usr/bin/env python3
# coding: utf-8

from src.valid_jargs_class import ValidJargs
from src.wrapper_class import Wrapper
from src.utilities import get_args

def main():
    args = get_args()
    Jargs = ValidJargs(args.parameter_json)
    if not args.dryrun:
        Cabinet = Wrapper(Jargs)
        Cabinet.run_wrapper()

if __name__ == "__main__":
    main()
