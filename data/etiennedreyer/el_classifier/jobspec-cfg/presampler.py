# IMPORT PACKAGES AND FUNCTIONS
import numpy     as np, multiprocessing, time, os, sys, h5py
from   argparse  import ArgumentParser
from   functools import partial
from   utils     import presample, merge_presamples


# OPTIONS
parser = ArgumentParser()
parser.add_argument( '--n_e'         , default = None        , type=float )
parser.add_argument( '--n_tasks'     , default = 12          , type=int   )
parser.add_argument( '--n_files'     , default = 10          , type=int   )
parser.add_argument( '--input_file'  , default = ''                       )
parser.add_argument( '--output_file' , default = 'el_data.h5'             )
parser.add_argument( '--sampling'    , default = 'ON'                     )
parser.add_argument( '--merging'     , default = 'ON'                     )
parser.add_argument( '--endcap'      , default = 'OFF'                    )
args = parser.parse_args()


# DATAFILES DEFINITIONS
file_path   = '/opt/tmp/godin/el_data/2019-06-20/' if args.input_file == '' else args.input_file
if not os.path.isdir(file_path+'output'): os.mkdir(file_path+'output')
output_path = file_path + 'output/'
data_files  = [file_path+h5_file for h5_file in os.listdir(file_path) if '.h5' in h5_file]
data_files  = sorted(data_files)[0:args.n_files]


# MERGING FILES/ NO PRESAMPLING
if args.sampling != 'ON' and args.merging == 'ON':
    temp_files = [output_path+h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file]
    sum_e      = [len(h5py.File(h5_file, 'r')['mcChannelNumber']) for h5_file in temp_files]
    print(); merge_presamples(sum_e[0], len(temp_files), output_path, args.output_file)
    print(sum(sum_e), 'ELECTRONS COLLECTED\n'); print(); sys.exit()
if args.sampling != 'ON' and args.merging != 'ON': sys.exit()


# ELECTRONS VARIABLES
images  =  ['em_barrel_Lr0'   , 'em_barrel_Lr1'   , 'em_barrel_Lr2'   , 'em_barrel_Lr3'    ,
            'tile_barrel_Lr1' , 'tile_barrel_Lr2' , 'tile_barrel_Lr3'                                     ]
tracks  =  ['tracks_pt'       , 'tracks_phi'      , 'tracks_eta'      , 'tracks_d0'        , 'tracks_z0'  ,
            'p_tracks_pt'     , 'p_tracks_phi'    , 'p_tracks_eta'    , 'p_tracks_d0'      , 'p_tracks_z0',
            'p_tracks_charge' , 'p_tracks_vertex' , 'p_tracks_chi2'   , 'p_tracks_ndof'    ,
            'p_tracks_pixhits', 'p_tracks_scthits', 'p_tracks_trthits', 'p_tracks_sigmad0'                ]
scalars =  ['p_truth_pt'      , 'p_truth_phi'     , 'p_truth_eta'     , 'p_truth_E'        , 'p_et_calo'  ,
            'p_pt_track'      , 'p_Eratio'        , 'p_phi'           , 'p_eta'            , 'p_e'        ,
            'p_Rhad'          , 'p_Rphi'          , 'p_Reta'          , 'p_d0'             , 'p_d0Sig'    ,
            'p_sigmad0'       , 'p_dPOverP'       , 'p_f1'            , 'p_f3'             , 'p_weta2'    ,
            'p_TRTPID'        , 'p_deltaEta1'     , 'p_LHValue'       , 'p_chi2'           , 'p_ndof'     ,
            'p_ECIDSResult'   , 'p_deltaPhiRescaled2'                                                     ]
integers = ['p_TruthType'     , 'p_iffTruth'      , 'p_nTracks'       , 'mcChannelNumber'  , 'eventNumber',
            'p_LHTight'       , 'p_LHMedium'      , 'p_LHLoose'       , 'p_numberOfSCTHits', 'p_charge'   ,
            'p_TruthOrigin'   , 'p_firstEgMotherTruthType', 'p_firstEgMotherTruthOrigin'                  ]

if args.endcap != 'OFF':
      images = ['em_endcap_Lr0'  , 'em_endcap_Lr1'  , 'em_endcap_Lr2'  , 'em_endcap_Lr3'    ,
              'lar_endcap_Lr0' , 'lar_endcap_Lr1' , 'lar_endcap_Lr2' , 'lar_endcap_Lr3'    ]
      tracks = []

# REMOVING TEMPORARY FILES (if any)
temp_files = [h5_file for h5_file in os.listdir(output_path) if 'temp' in h5_file]
for h5_file in temp_files: os.remove(output_path+h5_file)


# STARTING SAMPLING AND COLLECTING DATA
n_tasks = min(multiprocessing.cpu_count(), args.n_tasks)
max_e   = [len(h5py.File(h5_file, 'r')['train']['mcChannelNumber']) for h5_file in data_files]
if args.n_e != None: max_e = [min(np.sum(max_e),int(args.n_e))//(len(data_files))] * len(data_files)
max_e   = np.array(max_e)//n_tasks*n_tasks; sum_e = 0
print('\nSTARTING ELECTRONS COLLECTION (', '\b'+str(sum(max_e)), end=' ', flush=True)
print('electrons from', len(data_files),'files, using', n_tasks,'threads):')
pool = multiprocessing.Pool(n_tasks, maxtasksperchild=1)
for h5_file in data_files:
    batch_size = max_e[data_files.index(h5_file)]//n_tasks; start_time = time.time()
    print('Collecting', batch_size*n_tasks, 'e from:', h5_file.split('/')[-1], end=' ... ', flush=True)
    func   = partial(presample, h5_file, output_path, batch_size, sum_e, images, tracks, scalars, integers)
    sample = pool.map(func, np.arange(n_tasks))
    sum_e += batch_size
    print ('(', '\b'+format(time.time() - start_time,'.1f'), '\b'+' s)')
pool.close(); pool.join()


# MERGING FILES
if args.merging=='ON': merge_presamples(sum_e, n_tasks, output_path, args.output_file); print()
