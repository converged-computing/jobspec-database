import numpy as np
from modules.domain_reader import AstropyDomainReader   as Reader
from modules.domain_reader import DomainData
from modules.nht.source_finder import LikelihoodFinder as Finder
from modules.util.source_candidate import merge_source_candidates, remove_suspicious_candidates

from pickle import load

from pathlib import Path
import sys
import configparser

#===== domain indexing and config file  =====
n_domains = 1
domain_index = 0

if len(sys.argv) > 4:
    print("Program does not understand extra arguments. Expected input:\npython pipeline_full.py {config_file} {n_domains} {domain_index} ")
    sys.exit()
elif len(sys.argv) > 3:
    config_file = sys.argv[1]
    n_domains = int(sys.argv[2])
    domain_index = int(sys.argv[3])
    assert n_domains > domain_index
else:
    print("No command line arguments received to override domain numbering.")

####### READ USER CONFIG #######
config = configparser.ConfigParser()
config.read(config_file)

config = config['DEFAULT']
domain_border        = config.getint('domain_border')
n_domains_config     = config.getint('NDomains')
input_datacube       = config['input_datacube']
input_continuumimage = config['input_continuumimage']
output_directory     = config['output_directory']

square_domains    = config.getboolean('square_domains')
wavelet_threshold = config.getfloat('wavelet_threshold')
wavelet_zstep     = config.getint('wavelet_zstep')
lkh_threshold     = config.getfloat('lkh_threshold')
lkh_zbinning      = config.getint('lkh_zbinning')
lkh_isdatadenoised= config.getboolean('lkh_isdatadenoised') 
####### END USER CONFIG #######

assert n_domains == n_domains_config

#===== output file name templates  =====
denoised_file_name          = "{1}/{0}domains/sky_denoised3D_domain-{2}".format(        n_domains,    output_directory, domain_index)
source_candidates_file_name = "{1}/lkh_sources_th{2}_{0}.txt".format(domain_index, output_directory, lkh_threshold)
merged_source_candidates_file_name = "{1}/merged_sources_th{2}_{0}.txt".format(domain_index, output_directory, lkh_threshold)
output_catalog              = "{1}/catalog_new2_th{2}_{0}.txt".format(    domain_index, output_directory, lkh_threshold)

#===== create necessary directories  =====
Path(output_directory).mkdir(parents=True, exist_ok=True)
Path(denoised_file_name).mkdir(parents=True, exist_ok=True)

######################################################
# Step 1: Define domains
######################################################

print("====== Domain {0}: starting ======".format(domain_index ))

#===== define domains  =====

# interface to original data
reader_original = Reader(n_domains, domain_index, input_datacube, border = domain_border,  forceSquare=square_domains)
reader_original.define_domains()
reader_original._read_header()

#interface to denoised data
reader_denoised = Reader(n_domains, domain_index, input_datacube, border = domain_border,  forceSquare=square_domains)
reader_denoised.define_domains()
reader_denoised._read_header()

#print("I am domain {0} of {1}".format(domain_index+1, n_domains), flush=True)
print(reader_original)

######################################################
# Step 2: Denoise
######################################################

# define output file name
denoised_file_name += "_x-{0}-{1}_y-{2}-{3}_border{4}.npy".format(reader_original.xmin, reader_original.xmax, reader_original.ymin, reader_original.ymax, reader_original.border)

if Path(denoised_file_name).is_file():
    print ("====== Domain {0}: Denoised file {1} already exists, reading file ======".format(domain_index, denoised_file_name))
    reader_denoised.HI_cube = DomainData(infile = denoised_file_name)
else:
    print("====== Domain {0}: generating denoised domain {1} ======".format(domain_index,denoised_file_name ))
    # read the domain into memory
    reader_denoised.read()

    # denoise the domain
    from modules.denoiser import Denoiser2D1D as Denoiser
    denoiser = Denoiser(correlated_noise=False) 

    # allocate denoised cube
    output_cube = np.zeros(reader_denoised.HI_cube.data.shape)

    # we split the denoising into chunks along the z dimension, denoise in pieces, then combine them
    steps = np.linspace(0,reader_denoised.HI_cube.data.shape[0],wavelet_zstep, dtype = np.int)
    for k0, k1, k2, k3 in zip(np.append(steps[0],steps[:-2]), steps[:-1], steps[1:], np.append(steps[2:],steps[-1])):
        input_image = reader_denoised.HI_cube.data[k0:k3,:,:]
        denoised_cube = denoiser.denoise( np.copy(input_image), threshold_level=wavelet_threshold)#, num_scales_2d = 5, num_scales_1d = 5)
        output_cube[k1:k2,:,:] = denoised_cube[k1-k0:k2-k0,:,:]
    reader_denoised.HI_cube.data = output_cube
    reader_denoised.HI_cube.write(denoised_file_name)

######################################################
# Step 3: Find sources
######################################################

if Path(source_candidates_file_name).is_file():
    print ("====== Domain {0}: Source candidate file {1} already exists, reading file ======".format(domain_index, source_candidates_file_name),flush=True)
else:
    print("====== Domain {0}: finding candidate sources... ======".format(domain_index,source_candidates_file_name ), flush=True)
    from modules.nht.noise_profile import Landau, Gauss
    # configure NHT finder
    finder = Finder(reader_denoised, lkh_zbinning, Landau() if lkh_isdatadenoised else Gauss(), out_dir = output_directory, method="likelihood", df=20, lkh_threshold_attempts=10)
    # override algorithm's choice of likelihood threshold with user-defined choice
    finder._log_lkh_threshold = lkh_threshold

    source_candidates = finder.find_sources()
    print("Domain {0}: found {1} candidate sources.".format(domain_index,len(source_candidates) ), flush=True)
    with open(source_candidates_file_name, 'w') as f:
        for s in source_candidates: f.write(s.to_line())
print("====== Domain {0}: sources found, now merging close-by sources ======".format(domain_index), flush=True)

# merge sources using island-finding algorithm
merge_source_candidates(source_candidates_file_name, merged_source_candidates_file_name)

######################################################
# Step 4: Filter & Characterize
######################################################
print("====== Domain {0}: setting up Neural Network ======".format(domain_index), flush=True)
import tensorflow as tf
from modules.ai.utils import transforms, loss_fn, get_cutout
import modules.ai.classifier as classifier
import modules.ai.regressor as regressor

#===== set up CNN and input/output scale factors  =====
CNN_classifier=tf.keras.models.load_model("data/SKA_SDC2_classifier_CNN")
CNN_regressor =tf.keras.models.load_model("data/SKA_SDC2_regressor_CNN",
                                                custom_objects={'loss_fn': loss_fn})

input_sf = load(open('data/regressor_input_transform.pkl', 'rb'))

#===== classify sources  =====
print("====== Domain {0}: classifying sources ======".format(domain_index), flush=True)

if not reader_original.is_data_loaded():
    reader_original.read()

results = []
with open(merged_source_candidates_file_name, 'r') as f:
    for line in f:
        i, j, k, sig = line.split()
        i, j, k, sig = int(i), int(j), int(k), float(sig)
        ra, dec, freq = reader_original.pixels_to_sky(i,j,k)
 
        # get subcube around source posiiton
        cutout= get_cutout(i, j, k, reader_original, zwidth = 100)
        if cutout.shape[0] != 200:
            print("Unable to classify source with cutout dim", cutout.shape)
            continue
        cutout.shape = (1,*cutout.shape)

        # filter source based on classifier score
        classifier_score = CNN_classifier.predict(cutout*input_sf)
        if classifier_score[0][0] < 0.85: continue

        # predict source properties
        predictions  = CNN_regressor.predict( cutout*input_sf)

        # transform outputs
        predictions_tr = transforms.inv_transform(predictions)
        flux, hisize, sinposa, cosposa, inca, w20 = predictions_tr[0]
        posa = (-1*np.arctan2(sinposa,cosposa)*180/3.14159+360+90)%360

        result = "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(ra, dec, hisize, flux, freq, posa, inca, w20)
        results.append(result)
print("====== Domain {0}: writing final catalog ======".format(domain_index), flush=True)
#write final catalog
with open(output_catalog, 'w') as f:
    for r in results:
        f.write(r) 
print("====== Domain {0}: finished. ======".format(domain_index), flush=True)   

        

