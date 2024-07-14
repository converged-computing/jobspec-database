import os
from dotenv import load_dotenv
import numpy as np
from utils.mock_jet_data_helper import get_pulse
from data_transform import convert_pulse_dict_to_numpy_array

# Load .env variables
load_dotenv()

# Define basic inputs and metadata
PULSE_LIST = [81768,81798,85306,92207, 95479]
SIGNAL_DICT = {
    'efit': ['Rgeo', 'ahor', 'Vol', 'delRoben', 'delRuntn', 'k'],
    'power': ['P_OH', 'PNBI_TOT', 'PICR_TOT'], 
    'magn': ['IpiFP', 'BTF', 'q95'],
    'gas': ['D_tot'],
    'hrts': ['radius', 'ne', 'ne_unc', 'Te', 'Te_unc']  
}

# Prepare otuput directories for data and metadata
DATA_DIR = os.path.join(os.getcwd(), 'tmp/out')
os.makedirs(DATA_DIR, exist_ok=True)

# Execute data tranformations
for pulse_id in PULSE_LIST:
    pulse = get_pulse(pulse_id, SIGNAL_DICT)
    if pulse is None:
        print(f'Skipping {pulse_id}')
        continue
    pulse_transformed = convert_pulse_dict_to_numpy_array(pulse)
    for key, value in pulse_transformed.items():
        result_path = os.path.join(DATA_DIR, f'{pulse_id}_{key}.npy')
        print(f'Saving {result_path}')
        np.save(result_path, value)
