import os
import argparse
import math
import map_compression_and_plot


parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("--states", help="determines which states to look at")

args = parser.parse_args()

if args.states:
    states = args.states.split(',')
elif args.south == 'rim':
    states = ['MD', 'VA', 'NC', 'TN']
elif args.south == 'deep':
    states = ['AL', 'GA', 'LA', 'MS', 'SC']  
elif args.south == 'south':
    states = ['MD', 'VA', 'NC', 'TN','AL', 'GA', 'LA', 'MS', 'SC']  
else:
    # Open a file
    path = "outputs/bg/"
    states_dir = os.scandir(path)
    states = [state.name for state in states_dir]
    states.sort()



for state in states:
    path = f"outputs/bg/{state}/"
    state_dir = os.scandir(path)
    runs = [run.name[:-4] for run in state_dir if run.name[-3:]=='csv' if run.name[-5] != 's']
    state_dir = os.scandir(path)
    maps = [run.name[:-18] for run in state_dir if run.name[-7:]=='geojson']
    print(runs[:10])
    print('completed')
    print(maps)
    print('num_runs_left:' + f'{len([run for run in runs if run not in maps])}')    
    for run in runs:
        if run not in maps:
        #fname = path+run+'.geojson'
            print(run)
            
            #state = args.prefix[0:2]
            #objective = args.prefix[2:]
        
        
            csv_file = path + run + '.csv'
        
            shape_file = f'data2020/shapefiles/{state.lower()}_pl2020_bg_shp/{state.lower()}_pl2020_bg.shp'

            additional_data = [f'data2020/partisan_data/bg/{state.lower()}_bg_census_2020_voter_data_2020.csv', 
                       f'data2020/partisan_data/bg/{state.lower()}_bg_census_2020_voter_data_2016.csv']


            map_compression_and_plot.compress_and_plot(state,  csv_file, shape_file, additional_data)
    
    
