import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
import pandas as pd
# pytorch_fid was manually cloned into the project
from libs.pytorch_fid.src.fid.fid_score import calculate_fid_given_paths

HPC_PATH = "/gpfs/scratch/hebal100"
DATA_PATH = sys.argv[1] 
# directories where images of certain merge volume are stored
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']
m_vols = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
perf_values = []

for m_vol, dir in zip(m_vols, directories):
    fid = calculate_fid_given_paths((f'{HPC_PATH}/{DATA_PATH}/images_0', 
                                     f'{HPC_PATH}/{DATA_PATH}/{dir}'), batch_size=50, device='cuda', dims=2048)
    df = pd.read_csv(f'{HPC_PATH}/{DATA_PATH}/img_log.csv')
    df = df[df.m_vol == m_vol]
    avg_time = df['time'].values.mean()
    perf_values.append([m_vol, fid, avg_time])

perf_log = pd.DataFrame(perf_values, columns=['m_vol', 'fid', 'time'])
perf_log.to_csv(f'{HPC_PATH}/{DATA_PATH}/performance_log.csv', index=False)

