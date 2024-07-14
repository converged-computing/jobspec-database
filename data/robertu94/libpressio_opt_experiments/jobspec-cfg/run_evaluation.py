#!/usr/bin/env python
from typing import List, Any
from pathlib import Path
from pprint import pprint
import subprocess as sp
import argparse


parameters = ['size:size:compression_ratio', 'error_stat:error_stat:psnr',
              'ks_test:ks_test:pvalue', 'spatial_error:spatial_error:percent',
              'pearson:pearson:r']
tolerances = [30, .05, 5, .99999]
checks = ['<', '<', '>', '<']


def make_objective(prefix: str, tolerances: List[float], parameters: List[str]) -> str:

    tolerances_str = ", ".join(str(i) for i in tolerances)
    parameters_str = ", ".join(f"metrics['{prefix}/composite/{parameter}']" for parameter in parameters)
    check_str =  ", ".join(f'function(x,y) return x {check} y end' for check in checks)
    assert len(tolerances) == len(parameters) -1, "there must be one more parameter than tolerance"
    assert len(checks) == len(parameters) -1, "there must be one more parameter than checks"

    return f'''{prefix}/composite:composite:scripts=
       local parameters =  {{ {parameters_str} }}
       local N = {len(parameters)}
       local threshold = {{ {tolerances_str} }}
       local objective = 0;
       local error = parameters[1] == nil
       local checks = {{ {check_str} }}

       for i =2,N do
        if parameters[i] == nil then
            error = true
            break
        end
        if checks[i-1](parameters[i], threshold[i - 1]) then
            error = true
            break
        end
       end

       if error then
         objective = 0;
       else
         objective = parameters[1];
       end
       return 'objective', objective'''


def metrics_args(prefix: str) -> List[str]:
    """formats a list of metrics arguments for the current prefix

    prefix is expected to be a string of the form "/pressio" or "/pressio/sz"
    """
    plugins = ['time', 'size', 'error_stat', 'spatial_error',  'ks_test', 'pearson']
    args = []
    args.append("-b")
    args.append(prefix + ":" + prefix.split('/')[-1] + ":metric=composite")
    for plugin in plugins:
        args.append('-b')
        args.append(f'{prefix}/composite:composite:plugins={plugin}')
        args.append('-b')
        args.append(f'{prefix}/composite:composite:names={plugin}')
    args.append('-o')
    args.append(make_objective(prefix, tolerances, parameters))

    return args


def array_args(arg: str, values: List[Any], early=False) -> List[str]:
    """return an array of values expaneded into multiple cmdline arguments
    arg -- name of the argument
    values -- the array of values
    early -- whether to pass the argument as an "early argument" (-b) or not (-o).
    """
    args = []
    for v in values:
        args.append("-b" if early else "-o")
        args.append(f"{arg}={v!s}")
    return args


def build_pressio(args):
    DATALOAD = ['-i', '/zfs/fthpc/common/sdrbench/hurricane/CLOUDf48.bin',
                '-d', '100', '-d', '500', '-d', '500', '-t', 'float']
    SHOW_ARGS = ["-O", "all"] if args.show_args else []

    cmd_args = [
        "pressio", "-Q",  "-M", "all", *SHOW_ARGS,
        *DATALOAD,
        *metrics_args("/pressio"),
        *metrics_args("/pressio/sz"),
        "-b", "/pressio:opt:compressor=sz",
        "-b", "/pressio:opt:search=dist_gridsearch",
        "-b", "/pressio/dist_gridsearch:dist_gridsearch:search=guess_midpoint",

        "-o", "/pressio/dist_gridsearch:opt:target=60",
        "-o", "/pressio/dist_gridsearch:opt:global_rel_tolerance=.1",
        "-o", "/pressio/sz:sz:error_bound_mode_str=abs",
        "-o", "/pressio:opt:objective_mode_name=max",
        "-o", "/pressio/sz/composite/spatial_error:spatial_error:threshold=1e-4",
        "-o", "/pressio/composite/spatial_error:spatial_error:threshold=1e-4",
        *array_args("/pressio/dist_gridsearch:dist_gridsearch:num_bins", [args.nbins]),
        *array_args("/pressio/dist_gridsearch:dist_gridsearch:overlap_percentage",
                    [.1]),
        *array_args("/pressio/dist_gridsearch:opt:lower_bound", [1e-18]),
        *array_args("/pressio/dist_gridsearch:opt:upper_bound", [1e-12]),
        *array_args("/pressio:opt:inputs", ["/pressio/sz:sz:abs_err_bound"]),
        *array_args("/pressio:opt:output",
                    ["/pressio/sz/composite:composite:objective",
                     "/pressio/sz/composite/error_stat:error_stat:psnr",
                     "/pressio/sz/composite/size:size:compression_ratio",
                     "/pressio/sz/composite/ks_test:ks_test:pvalue",
                     "/pressio/sz/composite/spatial_error:spatial_error:percent",
                     "/pressio/sz/composite/pearson:pearson:r2"
                     ]),
        ]

    if not args.search_progress:
        cmd_args += [ "-b", "/pressio:opt:search_metrics=composite_search",]
    if args.debug:
        cmd_args = ["konsole", '-e', "gdb", "-x", "debug.gdb", "--args"] + cmd_args
    if args.parallel:
        cmd_args.insert(0, "mpiexec")
        if args.n_proc:
            cmd_args.insert(1, "-np")
            cmd_args.insert(2, str(args.n_proc))

    cmd_args.append("opt")
    return cmd_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_progress", action="store_true")
    parser.add_argument("--show_args", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_parallel", action="store_false", dest="parallel")
    parser.add_argument("--n_proc", type=int)
    parser.add_argument("--nbins", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cmd = build_pressio(args)
    if args.dry_run:
        pprint(cmd, compact=True)
    else:
        sp.run(cmd)

