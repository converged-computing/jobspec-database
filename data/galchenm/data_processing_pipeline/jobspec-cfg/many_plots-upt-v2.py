#!/usr/bin/env python3

"""
This script allows to plot in one figure two types of statistic files.
It is necessary to note, that before execution this script, you have to open Rsplit file and add one space before Rsplit column name.
Moreover, be careful with the order of files, because it is autumatically set color according their appearance:
if you want to have the same color for CC* plot and Rsplit plot files shoulde be placed at the same order:

CCstar1.dat CCstar2.dat CCstar3.dat ... Rsplit1.dat Rsplit2.dat Rsplit3.dat etc.
Additionally, the nubmer of colors is limited by ['b', 'g', 'r', 'c', 'm', 'y', 'k'], if you want more files,
please change the line: set_of_colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

You are able to plot with adding limit to y-axis if you have a prior knowledge.


python3 many_plots-upt-v2.py -i p8snr5_CCstar.dat p8snr8_CCstar.dat -x '1/d centre' -y 'CC*' -s 100.0 -o tmp3.png -add_nargs p8snr5_Rsplit.dat p8snr8_Rsplit.dat -yad 'Rsplit/%'

"""
import os
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from itertools import groupby, cycle 
from cycler import cycler

os.nice(0)

class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass


def parse_cmdline_args():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter)

    parser.add_argument('-i', nargs='+', type=str, help="List of files with the same name of columns for their comparing")
    parser.add_argument('-x', type=str, help="Name of x axis, for instance, '1/d centre'")
    parser.add_argument('-y', type=str, help="Name of y axis, for instance, CC*")
    parser.add_argument('-t', type=str, help="Title")
    parser.add_argument('-legend', nargs='+', type=str, help="Legend")
    parser.add_argument('-s', '--scale', type=float, help="Scale koefficient for y-axis")

    parser.add_argument('-x_lim_up','--x_limit_up', type=float, help="Limit value for x axis")
    parser.add_argument('-x_lim_dw','--x_limit_dw', type=float, help="Limit value for x axis")

    parser.add_argument('-o','--o', type=str, help="Name of output png file")

    parser.add_argument('-add_nargs', '--nargs_list_of_files', nargs='*', type=str, help="List of files for their comparing")
    parser.add_argument('-xad', '--x_add', type=str, help="Name of additional x axis, for instance, 1/d*, if you want to plot on the same picture")
    parser.add_argument('-yad', '--y_add', type=str, help="Name of additional y axis, for instance, CC*, if you want to plot on the same picture")

    parser.add_argument('-sad', '--scale_ad', type=float, help="Scale koefficient for y-axis")

    parser.add_argument('-l', '--logscale', type=bool, help="Use log scale or not on y axis")
    parser.add_argument('-r', '--reverse', type=bool, help="Use reverse x axis")
    parser.add_argument('-hor', '--horizontal',nargs='+', type=float, help="Value/s for horizontal line/s")
    parser.add_argument('-ver', '--vertical',nargs='+', type=float, help="Value/s for vertical line/s")
    parser.add_argument('-d', '--d', default=False, action='store_true', help="Use this flag if you want to show plot")
    return parser.parse_args()
  

def get_xy(file_name, x_arg_name, y_arg_name):
    x = []
    y = []

    with open(file_name, 'r') as stream:
        for line in stream:
            if y_arg_name in line:
                tmp = line.replace('1/nm', '').replace('# ', '').replace('centre', '').replace('/ A', '').replace(' dev','').replace('(A)','')
                tmp = tmp.split()
                y_index = tmp.index(y_arg_name)
                x_index = tmp.index(x_arg_name)

            else:
                tmp = line.split()
                x.append(float(tmp[x_index]))
                y.append(float(tmp[y_index]))

    
    x = np.array(x)
    y = np.array(y)
    
    list_of_tuples = list(zip(x, y))
    df = pd.DataFrame(list_of_tuples, 
                  columns = [x_arg_name, y_arg_name])
    
    df = df[df[y_arg_name].notna()]
    df = df[df[y_arg_name] > 0.]
    
    return df[x_arg_name], df[y_arg_name]



if __name__ == "__main__":
    args = parse_cmdline_args()
    input_files = args.i
    x_arg_name = args.x
    y_arg_name = args.y
    
    xxmin = 0.
    xxmax = -700000000000000000.
    
    yymin = 0.
    yymax = -700000000000000000.
    y_limit_down1 = 0
    y_limit_down2 = 0
    y_limit_up1 = -700000000000000000. 
    y_limit_up2 = -700000000000000000. 
    
    if len(input_files) > 8:
        set_of_colours = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
        '#17becf', '#9edae5']  
    else:
        set_of_colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k','o']
    
    colours_for_png = set_of_colours[:len(input_files)]
    current_path = os.getcwd()
    path_to_plots = os.path.join(current_path, 'plots_res')

    if not os.path.exists(path_to_plots):
        os.mkdir(path_to_plots)
    
    f1, ax = plt.subplots()
    cy = cycler('color', colours_for_png)
    ax2 = ax.twinx() 
    ax.set_prop_cycle(cy)
    ax2.set_prop_cycle(cy)


    for file_name in input_files:
        x, y = get_xy(file_name, x_arg_name, y_arg_name)
        
        xxmax = max(x) if max(x) > xxmax else xxmax
        
        
        if '%' in y_arg_name:
            new_y_arg_name = y_arg_name.split('/')[0]
        else:
            new_y_arg_name = y_arg_name
        if args.scale is not None:
            y *= args.scale
            new_y_arg_name = f'{args.scale}x{new_y_arg_name}'
        yymax = max(y) if max(y) > yymax else yymax
        y_limit_down1 = min(y) if min(y) < y_limit_down1 else y_limit_down1
        y_limit_up1 = yymax
        
        ax.plot(x, y, marker='.', label="{}({}) of {}".format(new_y_arg_name, x_arg_name, file_name))
   
    if args.nargs_list_of_files is not None:
        if args.x_add is not None:
            compare_x_arg_name = args.x_add
        else:
            compare_x_arg_name = x_arg_name

        if args.y_add is not None:
            compare_y_arg_name = args.y_add
        else:
            compare_y_arg_name = y_arg_name
            

        for file_name in args.nargs_list_of_files:
            compare_x, compare_y = get_xy(file_name, compare_x_arg_name, compare_y_arg_name)
            
            xxmax = max(compare_x) if max(compare_x) > xxmax else xxmax
            
            if '%' in compare_y_arg_name:
                new_compare_y_arg_name = compare_y_arg_name.split('/')[0]
            else:
                new_compare_y_arg_name = compare_y_arg_name
            if args.scale_ad is not None:
                compare_y *= args.scale_ad
                new_compare_y_arg_name = f'{scale_ad}x{new_compare_y_arg_name}'
            yymax = max(compare_y) if max(compare_y) > yymax else yymax
            y_limit_down2 = min(y) if min(y) < y_limit_down2 else y_limit_down2
            y_limit_up2 = yymax
            #ax.plot(compare_x, compare_y, marker='.', label="{}({}) of {}".format(new_compare_y_arg_name, compare_x_arg_name, file_name))
            ax2.plot(compare_x, compare_y, marker='.', label="{}({}) of {}".format(new_compare_y_arg_name, compare_x_arg_name, file_name))

    
    legends = args.legend

    
    if args.horizontal is not None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','o'][:len(args.horizontal)]
        ver = 1.2
        for hor,c in zip(args.horizontal,colors):
            ax.axhline(y=hor, xmin=xxmin, xmax=xxmax, linestyle='--', label=f'y={hor}', c=c)
            ax.text(ver, hor, f'x={ver}', horizontalalignment='center', fontweight='bold', color='black')
            ver+=0.4
            
    if args.vertical is not None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','o'][:len(args.vertical)]
        h = 15
        for ver,c in zip(args.vertical,colors):
            ax.axvline(x=ver, ymin=yymin, ymax=yymax, linestyle='--', label=f'x={ver}', c=c)
            ax.text(ver, h, f'x={ver}', rotation=90, verticalalignment='center', fontweight='bold', color='black')
            h+=10
    if args.logscale is not None:
        ax.set_yscale('log')

    if args.nargs_list_of_files is not None:
        ax.set_ylabel(new_y_arg_name) # + ' / ' + new_compare_y_arg_name)
        ax2.set_ylabel(new_compare_y_arg_name)
    else:
        ax.set_ylabel(new_y_arg_name)

    if args.x_limit_up is not None:
        x_limit_up = args.x_limit_up
    else:
        x_limit_up = 6.5

    if args.x_limit_dw is not None:
        x_limit_down = args.x_limit_dw
    else:
        x_limit_down = 0    
    
    y_limit_up1 = y_limit_up1 if y_limit_up1 <=100. else 110
    y_limit_up2 = y_limit_up2 if y_limit_up2 <=100. else 110

    y_limit_up1 = 1.1 if y_limit_up1 < 1. else y_limit_up1
    y_limit_up2 = 1.1 if y_limit_up2 < 1. else y_limit_up2
    
    ax.set_ylim(y_limit_down1, y_limit_up1)
    ax2.set_ylim(y_limit_down2, y_limit_up2)
    
    ax.set_xlim(x_limit_down, x_limit_up)
    ax.set_title(args.t)
    if args.reverse is not None:
        ax.invert_xaxis()
    
    ax.legend(legends, loc="center left")
    
    if x_arg_name == '1/d':
        ax.set_xlabel(x_arg_name + ' 1/nm')
    elif x_arg_name == 'd':
        ax.set_xlabel(x_arg_name + ' A')
    else:
        ax.set_xlabel(x_arg_name)
    print(os. getcwd())
    if args.o:
        os.chdir(path_to_plots)
        output = args.o
        f1.savefig(output)
        print(os. getcwd())
        #shutil.move(output, path_to_plots)
    
    if args.d:
        plt.show()