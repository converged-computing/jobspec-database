import pickle
import glob
import shutil
import os
import subprocess
from logger import log

def changeNucCenter(directory, xc, yc, zc):
    # Change center in Par_file_faults and output folder in PARAMETERS.par

    # Read PARAMETERS.par
    fPar = open("{0}/PARAMETERS.par".format(directory), 'r')
    par_file = fPar.read()
    fPar.close()

    # Change output folder
    par_file = par_file.format(xc=xc, yc=yc, zc=zc)

    # Write new PARAMETERS.par
    fPar = open("{0}/PARAMETERS.par".format(directory), 'w')
    fPar.write(par_file)
    fPar.close()

    # Create output folder
    if not os.path.exists("{0}/Results_xc{1}".format(directory, xc)):
        os.mkdir("{0}/Results_xc{1}".format(directory, xc))

    # Read in Par_file_faults
    fFault = open("{0}/Par_file_faults".format(directory), 'r')
    fault_file = fFault.read()
    fFault.close()

    # Change coordinates of center
    fault_file = fault_file.format(xc=xc, yc=yc, zc=zc)

    # Write new Par_file_faults
    fFault = open("{0}/Par_file_faults".format(directory), 'w')
    fFault.write(fault_file)
    fFault.close()


def vec_to_str_format(vec):
    res = "{0:>10.2f}".format(vec[0])
    for i in range(1, len(vec)):
        res += ' ' + "{0:>10.2f}".format(vec[i])
    return res

def listGoodFaultReceivers(faultreceiver_file, output_dir):

    # Read in the fault receivers
    fReceivers = open(faultreceiver_file, 'r')
    receivers = [[float(elem) for elem in line.split(' ')] for line in fReceivers.readlines()]
    fReceivers.close()

    # List all successful receivers in GoodFaultreceivers.dat
    fGoodReceivers = open(output_dir + "/GoodFaultreceivers.dat", 'w')
    for i in range(1, len(receivers) + 1):
        for filename in os.listdir(output_dir):
            if filename.startswith("data-faultreceiver-{0:05d}-".format(i)):
                fGoodReceivers.write("{0:>3d}: {1}\n".format(i, vec_to_str_format(receivers[i - 1])))
    fGoodReceivers.close()

if __name__ == '__main__':
    # Get directory names
    dirs_file = open("dirs.txt",'r')
    dirs = pickle.load(dirs_file)
    dirs_file.close()

    # For each mesh created
    oldDir = os.getcwd()
    for directory in dirs:

        log("--- Creating folder structure ---")
        # Copy the necessary files from setup
        log("Copying files from setup.")
        if os.path.exists(directory):
            shutil.rmtree(directory)
        shutil.copytree("{0}/setup/".format(oldDir), directory, symlinks=True)

        # Copy the mesh
        log("Copying mesh.")
        shutil.copytree("{0}_temp".format(directory), "{0}/mesh".format(directory), symlinks=True)
        shutil.rmtree("{0}_temp".format(directory))

        nuc_xcs = [0]
        for nuc_xc in nuc_xcs:

            log("Setting nucleation center to xc={0}.".format(nuc_xc))

            # Change the nucleation center
            changeNucCenter(directory, nuc_xc, 0, -10e3)

            # Run the simulation
            log("--- Starting SeisSol ---")
            os.chdir(directory)
            subprocess.call("mpiexec.hydra -n $mpi_ranks ./SeisSol_release_generatedKernels_dsnb_hybrid_none_9_4 PARAMETERS.par > Results_xc{0}/seissol.log".format(nuc_xc), shell=True)

            # Create list of successful fault receivers
            listGoodFaultReceivers("{0}/mesh/Faultreceiverlist.dat".format(directory), "{0}/Results_xc{1}".format(directory, nuc_xc))

    os.chdir(oldDir)
