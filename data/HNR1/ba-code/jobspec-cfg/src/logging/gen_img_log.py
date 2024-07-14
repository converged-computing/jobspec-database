import sys
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
import pandas as pd

HPC_PATH = "/gpfs/scratch/hebal100"
MAIN_DIR = sys.argv[1] 

# names of log files of run16
files = ['log_3fBc5.csv', 'log_BrAkV.csv', 'log_f77w7.csv', 'log_lqsyS.csv',
         'log_vq59J.csv', 'log_64YNq.csv', 'log_C4AiR.csv', 'log_IGnWZ.csv',
         'log_rV7Lh.csv', 'log_VWBJq.csv']

logs = pd.read_csv(f'{HPC_PATH}/{MAIN_DIR}/logger/{files[0]}')
for file in files[1:]:
    log = pd.read_csv(f'{HPC_PATH}/{MAIN_DIR}/logger/{file}')
    logs = pd.concat([logs, log], ignore_index=True)

logs.to_csv(f'{HPC_PATH}/{MAIN_DIR}/img_log.csv', index=False)
