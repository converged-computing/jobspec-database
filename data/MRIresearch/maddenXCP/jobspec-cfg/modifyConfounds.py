import os
import datetime
import numpy as np
import pandas as pd
from shutil import copyfile

def extract_confounds(confound, fields ):
    confound_df = pd.read_csv(confound, delimiter='\t')
    confound_vars=fields
    confound_df = confound_df[confound_vars]
    return confound_df

def isTrue(arg):
	return arg is not None and (arg == 'Y' or arg == '1' or arg == 'True') 

def get_parser():
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description="Artificially change the fd of volumes neighboring a motion outlier."
        "Pass in a confounds file from fmriprep and supply a framewise displacement threshold.",formatter_class=RawTextHelpFormatter)
    parser.add_argument('confound_file', action='store',
        help='Location/Name of the fmriprep confound file.')
    parser.add_argument('--confound_out', action='store', 
        help='Location/Name of backup for fmriprep confound file')
    parser.add_argument('--fd_name', action='store',type=str, default='framewise_displacement',
        help='The name of framewise displacement field in fmriprep confounds file. Defaults to framewise_displacement')
    parser.add_argument('--fd_thresh', action='store',type=float, default=0.5,
        help='Framewise displacement threshold. Defaults to 0.5mm.')
    parser.add_argument('--fd_replacement', action='store',type=float, default=0.51,
        help='Framewise displacement to insert in neighbouring volumes. defaults to 0.51.')
    parser.add_argument('--vols_before', action='store',type=float, default=1,
        help='Number of volumes to scrub before.')
    parser.add_argument('--vols_after', action='store',type=float, default=1,
        help='Number of volumes to scrub after.')
    parser.add_argument('--nan_as_zero',nargs='*', default='True',
        help='Replace NaNs in regressor with zero. This is he default. Alternative is to use the mean')
    return parser


def main():

    TIMESTAMP=datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    opts = get_parser().parse_args()

    # Replace nan as mean or zero?
    NAN_AS_ZERO = isTrue(opts.nan_as_zero)

    confound_file=os.path.abspath(opts.confound_file)
    if not(opts.confound_out == None):
        confound_out=os.path.abspath(opts.confound_out)
    else:
        confound_out=confound_file.split('task')[0]+'confoundfile.backup.' + TIMESTAMP

    # back up confound file
    copyfile(confound_file,confound_out)

    if opts.fd_name:
        fd_name=opts.fd_name

    if opts.fd_thresh:
        fd_thresh=opts.fd_thresh

    if opts.fd_replacement:
        fd_replacement=opts.fd_replacement

    if opts.vols_before:
        vols_before=opts.vols_before

    if opts.vols_after:
        vols_after=opts.vols_after

    confounds = extract_confounds(confound_file,
                                    fd_name)
    

    # fill NaN with mean or zero
    if (NAN_AS_ZERO):
        confounds= confounds.fillna(0)
    else:
        confounds= confounds.fillna(np.mean(confounds))

    confounds_array=confounds.to_numpy()
    timespan=len(confounds_array)

    outliers=np.argwhere(confounds_array >= fd_thresh)
   
    CONFOUND_CHANGED=False
    new_fd=np.zeros(timespan)
    for outlier in outliers:
        outlier_ind=outlier[0]
        outlier_range=np.arange(outlier_ind - vols_before,outlier_ind + 1 + vols_after)
        adj_out = np.delete(outlier_range, np.where(outlier_range == outlier_ind))
        
        adj_out=adj_out.astype(int)

        # ensure no negative indices
        adj_out=adj_out[adj_out >= 0 ]
        # ensure we don't exceed number of volumes
        adj_out=adj_out[adj_out < timespan ]
        
        
        if len(adj_out) > 0:
            confounds_array[adj_out]=fd_replacement
            CONFOUND_CHANGED=True
    
    if CONFOUND_CHANGED:
        df = pd.DataFrame(data=confounds_array.flatten())
        confound_df = pd.read_csv(confound_file, delimiter='\t')
        confound_df[fd_name]=df
        confound_df.to_csv(confound_file,index=False,header=True,sep="\t")
    
    

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
