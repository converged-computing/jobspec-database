import sys
import os
import subprocess
import h5py
import time
import numpy as np
import argparse
import multiprocessing
from functools import partial
import psutil




# -----------------------------
# -----------------------------
# --- Command Line Arguments --
# -----------------------------
# -----------------------------


class CommandLineArgs:

    def __init__(self):
        eval_parser = argparse.ArgumentParser(
            description="Parareal code",
            fromfile_prefix_chars="@",
            allow_abbrev=False,
            epilog="It's more Para than Real...",
        )
        eval_parser.add_argument(
            "-no_ref",
            "--no_reference_run",
            action='store_true',
            help="Do not run the fine solver reference run.",
        )
        eval_parser.add_argument(
            "-np",
            "--para_chunks",
            action="store",
            type=int,
            help="Specify the number of Parareal time chunks",
        )
        eval_parser.add_argument(
            "-npmax",
            "--para_max_iter",
            action="store",
            type=int,
            help="Specify the maximum number of parareal cycle",
        )
        eval_parser.add_argument(
            "-acc",
            "--para_accuracy",
            action="store",
            type=float,
            help="Specify the convergence accuracy for Parareal",
        )
        eval_parser.add_argument(
            "-ip",
            "--iter_para",
            action="store",
            nargs='?',
            const=0,
            default=0,
            type=int,
            help="Specify a restart iteration of the parareal cycle",
        )
        eval_parser.add_argument(
            "-ic",
            "--iter_chunk",
            action="store",
            nargs='?',
            const=0,
            default=0,
            type=int,
            help="Specify a restart iteration of the time-chunk solver within a parareal cycle.",
        )
        eval_parser.add_argument(
            "-ninn",
            "--n_input_nn",
            action="store",
            nargs='?',
            const=0,
            default=0,
            type=int,
            help="Specify the number of input t-steps for the NN solver.",
        )
        eval_parser.add_argument(
            "-nonn",
            "--n_output_nn",
            action="store",
            nargs='?',
            const=0,
            default=0,
            type=int,
            help="Specify the number of output t-steps for the NN solver.",
        )
        eval_parser.add_argument(
            "-chkpt",
            "--checkpoint_dir",
            action="store",
            type=self._dir_path,
            help="set the directory from which to start the whole Parareal run (instead of running the initial grid)",
        )
        eval_parser.add_argument(
            "-multi_chkpt",
            "--multi_chkpt",
            action='store_true',
            help="If True, then the parareal corrector/predictors will be applied to all the available checkpoint files, not just the last one.",
        )
        eval_parser.add_argument(
            "-coarse_not_jorek",
            "--coarse_not_jorek",
            action='store_true',
            help="If True, then it means the coarse solver is not JOREK itself (eg. ML surrogate) and we assume its output data is already in right format (ie. no FEM projection/extraction needed).",
        )
        eval_parser.add_argument(
            "-coarse_not_slurm",
            "--coarse_not_slurm",
            action='store_true',
            help="If True, then we will not submit the coarse runs to slurm queue, they will be run interactively, as part of this script.",
        )

        self.args = eval_parser.parse_args()
        print("Command Line Arguments: ", vars(self.args))
        sys.stdout.flush()

        if (self.args.para_chunks == None):
            self.args.para_chunks = 10
            print("Warning: You did not specify a number of Parareal time-chunks with the -np option.")
            print("         Defaulting to %d." % (self.args.para_chunks) )
        if (self.args.para_max_iter == None):
            self.args.para_max_iter = self.args.para_chunks
            print("Warning: You did not specify a limit of Parareal iteration cycles with the -npmax option.")
            print("         Defaulting to %d." % (self.args.para_max_iter) )
        if (self.args.para_accuracy == None):
            self.args.para_accuracy = 1.e-3
            print("Warning: You did not specify a convergence accuracy for Parareal with the -acc option.")
            print("         Defaulting to %f." % (self.args.para_accuracy) )
        if self.args.coarse_not_slurm and (not self.args.coarse_not_jorek):
            print("Warning: If your coarse solver is JOREK, we strongly suggest to submit the runs to SLURM rather than run them interactivel with the flag -coarse_not_slurm")
        if (self.args.n_input_nn == 0):
            self.args.n_input_nn = 5
            print("Warning: You did not specify a number of input-steps for the NN coarse solver with the -ninn.")
            print("         Defaulting to %d." % (self.args.n_input_nn) )
        if (self.args.n_output_nn == 0):
            self.args.n_output_nn = 5
            print("Warning: You did not specify a number of output-steps for the NN coarse solver with the -nonn.")
            print("         Defaulting to %d." % (self.args.n_output_nn) )

    def _file_path(self, string):
        if os.path.isfile(string):
            return string
        else:
            print("-------------------------------------------")
            print("String: %s is not a file" % (string))
            print("-------------------------------------------")
            print("Please provide a valid filename")
            print("-------------------------------------------")
            sys.stdout.flush()
            raise FileNotFoundError

    def _dir_path(self, string):
        if os.path.isdir(string):
            return string
        else:
            print("-------------------------------------------")
            print("String: %s is not a directory" % (string))
            print("-------------------------------------------")
            print("Please provide a valid directory path")
            print("-------------------------------------------")
            sys.stdout.flush()
            raise FileNotFoundError




# -----------------------------
# -----------------------------
# --- Utils/Tools Functions ---
# -----------------------------
# -----------------------------


# --- Easy wrapper for command line execution, can return text output to print, and logs it to a logfile
class CommandLineExecution:

    def __init__(self, logfile):
        self.logfile = open(logfile, "w")

    def exec(self, commandline, verbose=False):
        text_output = subprocess.check_output(commandline, shell=True, text=True)
        # --- print if verbose required
        if (verbose):
            print(text_output) ; sys.stdout.flush()
        # --- Log output to file (always)
        self.logfile.write(text_output)
        # --- Return output text if needed by user
        return text_output





# --- Easy wrapper for file modifications (based on an old perl script)
class ChangeFile:

    def __init__(self, CMD, io_tools_dir):
        self.CMD = CMD # commandline tool
        self.change_file_script = os.path.join(io_tools_dir,"my_change_file.perl")
        if not os.path.exists(self.change_file_script):
            print('Something wrong while initialising ChangeFile, script "%s" does not exist, aborting...' % (self.change_file_script) ) ; sys.stdout.flush()
            sys.exit(0)

    def change_file(self, filepath, string, new_string, verbose=False):
        commandline = self.change_file_script + " -file "+filepath+" -string '"+string+"' -new '"+new_string+"'"
        self.CMD.exec(commandline, verbose)



# --- Generic naming of directories
class DirectoryNames:

    def __init__(self, cwd):
        self.cwd = cwd

    def para_dir_name(self, i_para):
        return "para_iter_"+str(i_para).zfill(3)

    def para_dir_path(self, i_para):
        return os.path.join(self.cwd, self.para_dir_name(i_para))

    def coarse_dir_name(self, i_chunk):
        return "run_coarse_"+str(i_chunk).zfill(3)

    def coarse_dir_path(self, i_para, i_chunk):
        return os.path.join(self.cwd, self.para_dir_name(i_para), self.coarse_dir_name(i_chunk))

    def fine_dir_name(self, i_chunk):
        return "run_fine_"+str(i_chunk).zfill(3)

    def fine_dir_path(self, i_para, i_chunk):
        return os.path.join(self.cwd, self.para_dir_name(i_para), self.fine_dir_name(i_chunk))

    # --- Check files exists and return relative location for sym-links
    def get_symlink_file(self, i_para, i_chunk, coarse_or_fine, previous_file):
        if (coarse_or_fine == "coarse"):
            check_file = os.path.join(self.coarse_dir_path(i_para,i_chunk),previous_file) # check the full path
            chkpt_dir  = os.path.join("..","..",self.para_dir_name(i_para),self.coarse_dir_name(i_chunk)) # we want the sym-link to be relative
            symlink    = os.path.join(chkpt_dir,previous_file)
        else:
            check_file = os.path.join(self.fine_dir_path(i_para,i_chunk),previous_file) # check the full path
            chkpt_dir  = os.path.join("..","..",self.para_dir_name(i_para),self.fine_dir_name(i_chunk)) # we want the sym-link to be relative
            symlink    = os.path.join(chkpt_dir,previous_file)
        if not os.path.exists(check_file):
            print("Warning: Could not find checkpoint file \n  %s,\naborting..." % (check_file) )
            sys.stdout.flush()
            sys.exit(0)

        return symlink, chkpt_dir





def safe_symlink(CMD, filename, symlink):
    CMD.exec("rm -f "+symlink)
    CMD.exec("ln -s "+filename+" "+symlink)




def get_all_checkpoint_files(CMD,run_dir,chkpt_dir,cwd,file_type):
    # --- Get all files assuming the standard JOREK output name
    command = "ls " + os.path.join(chkpt_dir,file_type)
    file_list = CMD.exec("cd "+run_dir+"; "+command+" ; cd "+cwd)
    file_list = file_list.split()
    str_isolate_name = file_type.split("0*.h5")[0]
    index_list = []
    for file_tmp in file_list:
        cut_str = file_tmp.split(str_isolate_name)[-1]
        index_list.append( cut_str.split(".h5")[0] )
    return file_list, index_list






# -----------------------------
# -----------------------------
# --- Parareal Predictor    ---
# -----------------------------
# -----------------------------



# --- MPI: Execution of grid interpolation for Parareal Predictor/Corrector, to be run in parallel
def MPI_parareal_predictor(data_broadcast,i_file):
    # --- Class to execute commandlines
    MPI_id = str(psutil.Process().cpu_num()).zfill(2)
    CMD = CommandLineExecution("cmd_logs"+MPI_id+".txt")
    # --- Retrieve broadcast data
    n_files      = data_broadcast['n_files']   
    index_list   = data_broadcast['index_list']
    files_list   = data_broadcast['files_list']
    choices_dict = data_broadcast['choices_dict']
    cwd          = data_broadcast['cwd']
    run_dir      = data_broadcast['run_dir']
    io_tools_dir = data_broadcast['io_tools_dir']
    index_tmp    = index_list[i_file]
    chkpt_tmp    = files_list[i_file]
    # --- Filename to be written
    data_out_filename = "jorek_data"+index_tmp+".h5"
    # --- Print MPI progression ?
    #print('MPI process %d out of %d to create file %s' % (i_file,n_files,data_out_filename) ) ; sys.stdout.flush()
    if ( choices_dict["coarse_run"] and choices_dict["coarse_not_jorek"] ):
        safe_symlink(CMD, files_list[i_file], os.path.join(run_dir,data_out_filename))
        if (i_file == n_files-1):
            safe_symlink(CMD,data_out_filename,os.path.join(run_dir,"jorek00000.h5"))
    else:
        safe_symlink(CMD,files_list[i_file],os.path.join(run_dir,"gauss_data"+index_tmp+".h5"))
        # --- project data
        command = "mpirun -np 1 "+os.path.join(cwd,io_tools_dir,"jorek2_parareal_project")+" < input -i gauss_data"+index_tmp+".h5 -j jorek_grid.h5 -o "+data_out_filename
        CMD.exec("cd "+run_dir+"; "+command+" ; cd "+cwd)
        if (i_file == n_files-1):
            safe_symlink(CMD,data_out_filename,os.path.join(run_dir,"jorek00000.h5"))
            CMD.exec("cd "+run_dir+"; cp jorek00000.h5 jorek_restart.h5 ; cd "+cwd)




def parareal_predictor(CMD,run_dir,chkpt_dir,cwd,io_tools_dir,restart_file, choices_dict):
    # --- Get list of files to which we need to apply predictor, only if using coarse run, since fine solver (JOREK itself) only needs the last checkpoint file
    if choices_dict["coarse_run"]:
        file_type = "jorek_extracted_coarse0*.h5"
    else:
        file_type = "jorek_extracted_fine0*.h5"
    files_list, index_list  = get_all_checkpoint_files(CMD,run_dir,chkpt_dir,cwd, file_type)
    if not (choices_dict["multi_chkpt"] and choices_dict["coarse_run"]):
        files_list = [ files_list[-1] ]
        index_list = [""]
    # --- Prepare parallelisation over all chkpt files
    # --- Data sent to each process (can't be too large, so can't include camera "data" itself)
    n_files = len(files_list)
    data_broadcast = {}
    data_broadcast['n_files']      = n_files
    data_broadcast['index_list']   = index_list
    data_broadcast['files_list']   = files_list
    data_broadcast['choices_dict'] = choices_dict
    data_broadcast['cwd']          = cwd         
    data_broadcast['run_dir']      = run_dir     
    data_broadcast['io_tools_dir'] = io_tools_dir
    # --- Functions specification for mpi processes
    parallel_function = partial(MPI_parareal_predictor, data_broadcast)
    parallel_array = [i for i in range(n_files)]
    # --- Set up MPI processes
    n_MPI = multiprocessing.cpu_count()
    n_processes = min(n_MPI,n_files)
    print('Predictor: Interpolating %d files in parallel using %d processes...' % (n_files,n_processes) ) ; sys.stdout.flush()
    with multiprocessing.Pool(n_processes) as mpi_pool:
        mpi_pool.map(parallel_function, parallel_array)







# ------------------------------------
# ------------------------------------
# --- Parareal Predictor-corrector ---
# ------------------------------------
# ------------------------------------



# --- MPI: Execution of grid interpolation for Parareal Predictor/Corrector, to be run in parallel
def MPI_parareal_predictor_corrector(data_broadcast,i_file):
    # --- Class to execute commandlines
    MPI_id = str(psutil.Process().cpu_num()).zfill(2)
    CMD = CommandLineExecution("cmd_logs"+MPI_id+".txt")
    # --- Retrieve broadcast data
    n_files                = data_broadcast['n_files']   
    index_list             = data_broadcast['index_list']
    files_list_coarse      = data_broadcast['files_list_coarse']
    files_list_coarse_prev = data_broadcast['files_list_coarse_prev']
    files_list_fine        = data_broadcast['files_list_fine']
    choices_dict           = data_broadcast['choices_dict']
    cwd                    = data_broadcast['cwd']
    run_dir                = data_broadcast['run_dir']
    io_tools_dir           = data_broadcast['io_tools_dir']
    index_tmp              = index_list[i_file]
    # --- Print MPI progression ?
    #print('MPI process %d out of %d for predictor/corrector step' % (i_file,n_files) ) ; sys.stdout.flush()

    # --- Filename to be written

    # ---------------------------------------------
    # --- First the coarse file from previous chunk
    # ---------------------------------------------
    if ( choices_dict["coarse_run"] and choices_dict["coarse_not_jorek"] ):
        safe_symlink(CMD,files_list_coarse[i_file],os.path.join(run_dir,"mesh_data_coarse"+index_tmp+".h5"))
    else:
        safe_symlink(CMD,files_list_coarse[i_file],os.path.join(run_dir,"gauss_data_coarse"+index_tmp+".h5"))

    # ------------------------------------------------------------------
    # --- Coarse file from previous chunk of previous Parareal iteration
    # ------------------------------------------------------------------
    if ( choices_dict["coarse_run"] and choices_dict["coarse_not_jorek"] ):
        safe_symlink(CMD,files_list_coarse_prev[i_file],os.path.join(run_dir,"mesh_data_coarse_prev"+index_tmp+".h5"))
    else:
        safe_symlink(CMD,files_list_coarse_prev[i_file],os.path.join(run_dir,"gauss_data_coarse_prev"+index_tmp+".h5"))

    # ----------------------------------------------------------------
    # --- Fine file from previous chunk of previous Parareal iteration
    # ----------------------------------------------------------------
    if ( choices_dict["coarse_run"] and choices_dict["coarse_not_jorek"] ):
        safe_symlink(CMD,files_list_coarse_prev[i_file],os.path.join(run_dir,"mesh_data_fine"+index_tmp+".h5"))
    else:
        safe_symlink(CMD,files_list_coarse_prev[i_file],os.path.join(run_dir,"gauss_data_fine"+index_tmp+".h5"))

    # ---------------------------------
    # --- Apply the predictor/corrector
    # ---------------------------------
    if ( choices_dict["coarse_run"] and choices_dict["coarse_not_jorek"] ):
        command = "python3 "+os.path.join(cwd,io_tools_dir,"predictor_corrector.py")+" -m -c mesh_data_coarse"+index_tmp+".h5  -cp mesh_data_coarse_prev"+index_tmp+".h5  -f mesh_data_fine"+index_tmp+".h5  -o mesh_data"+index_tmp+".h5"
    else:
        command = "python3 "+os.path.join(cwd,io_tools_dir,"predictor_corrector.py")+" -g -c gauss_data_coarse"+index_tmp+".h5 -cp gauss_data_coarse_prev"+index_tmp+".h5 -f gauss_data_fine"+index_tmp+".h5 -o gauss_data"+index_tmp+".h5"
    CMD.exec("cd "+run_dir+"; "+command+" ; cd "+cwd)

    # -------------------------------
    # --- Project data back onto grid
    # -------------------------------
    # --- project data
    if ( choices_dict["coarse_run"] and choices_dict["coarse_not_jorek"] ):
        command = "mv mesh_data"+index_tmp+".h5 jorek_data"+index_tmp+".h5"
        CMD.exec("cd "+run_dir+"; "+command+" ; cd "+cwd)
        if (i_file == n_files-1):
            safe_symlink(CMD,"jorek_data"+index_tmp+".h5",os.path.join(run_dir,"jorek00000.h5"))
        command = "rm -f mesh_data_coarse"+index_tmp+".h5 mesh_data_coarse_prev"+index_tmp+".h5 mesh_data_fine"+index_tmp+".h5"
        CMD.exec("cd "+run_dir+"; "+command+" ; cd "+cwd)
    else:
        command = "mpirun -np 1 "+os.path.join(cwd,io_tools_dir,"jorek2_parareal_project")+" < input -i gauss_data"+index_tmp+".h5 -j jorek_grid.h5 -o jorek_para"+index_tmp+".h5"
        CMD.exec("cd "+run_dir+"; "+command+" ; cd "+cwd)
        # --- copy data to checkpoint
        if (i_file == n_files-1):
            safe_symlink(CMD,"jorek_para"+index_tmp+".h5",os.path.join(run_dir,"jorek00000.h5"))
            CMD.exec("cd "+run_dir+"; cp jorek_para"+index_tmp+".h5 jorek_restart.h5 ; cd "+cwd)
        command = "rm -f gauss_data_coarse"+index_tmp+".h5 gauss_data_coarse_prev"+index_tmp+".h5 gauss_data_fine"+index_tmp+".h5"
        CMD.exec("cd "+run_dir+"; "+command+" ; cd "+cwd)







def parareal_predictor_corrector(CMD,run_dir,chkpt_dirs,cwd,io_tools_dir, choices_dict):
    # --- Get list of files to which we need to apply predictor, only if using coarse run, since fine solver (JOREK itself) only needs the last checkpoint file
    if choices_dict["coarse_run"]:
        file_type = "jorek_extracted_coarse0*.h5"
    else:
        file_type = "jorek_extracted_fine0*.h5"
    files_list_coarse,      index_list  = get_all_checkpoint_files(CMD,run_dir,chkpt_dirs["chkpt_dir_coarse"],cwd, file_type)
    files_list_coarse_prev, index_list2 = get_all_checkpoint_files(CMD,run_dir,chkpt_dirs["chkpt_dir_coarse_prev"],cwd, file_type)
    files_list_fine,        index_list3 = get_all_checkpoint_files(CMD,run_dir,chkpt_dirs["chkpt_dir_fine"],cwd, file_type)
    # --- Sanity check (note I'm worried this might fail at the first chunk of the 2nd cycle...)
    if not ( np.array_equal(index_list,index_list2) and np.array_equal(index_list2,index_list3) ):
        sys.exit("Warning! The Parareal predictor/corrector cannot find same number of files in all chkpt_dirs")
    if not (choices_dict["multi_chkpt"] and choices_dict["coarse_run"]):
        files_list_coarse      = [ files_list_coarse[-1] ]
        files_list_coarse_prev = [ files_list_coarse_prev[-1] ]
        files_list_fine        = [ files_list_fine[-1] ]
        index_list = [""]
    # --- Prepare parallelisation over all chkpt files
    # --- Data sent to each process (can't be too large, so can't include camera "data" itself)
    n_files = len(files_list_coarse)
    data_broadcast = {}
    data_broadcast['n_files']                = n_files
    data_broadcast['index_list']             = index_list
    data_broadcast['files_list_coarse']      = files_list_coarse     
    data_broadcast['files_list_coarse_prev'] = files_list_coarse_prev
    data_broadcast['files_list_fine']        = files_list_fine       
    data_broadcast['choices_dict']           = choices_dict
    data_broadcast['cwd']                    = cwd         
    data_broadcast['run_dir']                = run_dir     
    data_broadcast['io_tools_dir']           = io_tools_dir
    # --- Functions specification for mpi processes
    parallel_function = partial(MPI_parareal_predictor_corrector, data_broadcast)
    parallel_array = [i for i in range(n_files)]
    # --- Set up MPI processes
    n_MPI = multiprocessing.cpu_count()
    n_processes = min(n_MPI,n_files)
    print('Predictor: Interpolating %d files in parallel using %d processes...' % (n_files,n_processes) ) ; sys.stdout.flush()
    with multiprocessing.Pool(n_processes) as mpi_pool:
        mpi_pool.map(parallel_function, parallel_array)















# -----------------------------
# -----------------------------
# --- Main Program ------------
# -----------------------------
# -----------------------------




# --- Main program
def main():

    # -----------------------------
    # --- Run setup and inuts -----
    # -----------------------------

    print('JOREK Parareal program starting') ; sys.stdout.flush()

    # --- Get command line args
    cli = CommandLineArgs() ; sys.stdout.flush()
    n_time_chunks        = cli.args.para_chunks # number of Parareal time chunks
    n_iter_max           = cli.args.para_max_iter # Maximum number of parareal iterations
    accuracy             = cli.args.para_accuracy # accuracy level required for Parareal convergence (on density differences)
    restart_para         = cli.args.iter_para
    restart_chunk        = cli.args.iter_chunk
    checkpoint_dir       = cli.args.checkpoint_dir
    multi_chkpt          = cli.args.multi_chkpt
    coarse_not_jorek     = cli.args.coarse_not_jorek
    coarse_not_slurm     = cli.args.coarse_not_slurm
    n_input_nn           = cli.args.n_input_nn
    n_output_nn          = cli.args.n_output_nn
    if ( (restart_para != 0) or (restart_chunk != 0) ):
        print("Warning: restarting Parareal run at Parareal Cycle #%d and time-chunk #%d" % (restart_para,restart_chunk) )
    checkpoint_rst = False
    if ( (checkpoint_dir != None) and (checkpoint_dir != "") ):
        checkpoint_rst = True

    # --- Dictionary of some of the main user-choices that need to be communicated
    choices_dict = {"multi_chkpt": multi_chkpt,
                    "coarse_not_jorek": coarse_not_jorek,
                    "coarse_input": False,
                    "coarse_run": False
                   }

    # --- Class to execute commandlines
    CMD = CommandLineExecution("cmd_logs.txt")

    # --- Basic directories needed for the run
    coarse_solver_dir = "./coarse_solver"
    fine_solver_dir   = "./fine_solver"
    io_tools_dir      = "./io_tools"
    cwd               = os.getcwd()

    # --- Get the size of timesteps from coarse and fine solvers (may be different)
    tstep_fine   = CMD.exec("grep tstep "+os.path.join(fine_solver_dir,"input"))
    tstep_fine   = tstep_fine.split("=")[1]
    tstep_fine   = float(tstep_fine.split()[0])
    tstep_coarse = CMD.exec("grep tstep "+os.path.join(coarse_solver_dir,"input"))
    tstep_coarse = tstep_coarse.split("=")[1]
    tstep_coarse = float(tstep_coarse.split()[0])
    # --- Get the total number of timesteps from fine solvers (note: is this the right thing to assume?)
    nstep_fine   = CMD.exec("grep nstep "+os.path.join(fine_solver_dir,"input"))
    nstep_fine   = nstep_fine.split("=")[1]
    nstep_fine   = int(nstep_fine.split()[0])
    # --- Get the frequency of output checkpoints
    nout_fine   = CMD.exec("grep nout "+os.path.join(fine_solver_dir,"input"))
    nout_fine   = nout_fine.split("=")[1]
    nout_fine   = int(nout_fine.split()[0])
    nout_coarse = CMD.exec("grep nout "+os.path.join(coarse_solver_dir,"input"))
    nout_coarse = nout_coarse.split("=")[1]
    nout_coarse = int(nout_coarse.split()[0])
    # --- Check that the size of time steps is coherent between coarse and fine solvers
    if (tstep_coarse%tstep_fine > 1.e-12):
        print("Warning: You need to make sure that the timestep in ./coarse_solver/input is")
        print("a round multiple of the one in ./fine_solver/input")
        print("Aborting...")
        sys.exit(1)
    tstep_fract = int(tstep_coarse / tstep_fine)
    print("Total number of time-steps in the run: %d" % (nstep_fine) )
    print("Size of time-steps in fine   solver: %f" % (tstep_fine) )
    print("Size of time-steps in coarse solver: %f" % (tstep_coarse) )
    # --- Get number of steps for each chunk
    n_time_total  = nstep_fine # total number of steps for the entire run
    if (n_time_total % n_time_chunks != 0):
        print("Please make sure the total number of time-steps %d is divisible by the number of Parareal time chunks %d, aborting..." % (n_time_total,n_time_chunks) )
        sys.exit(0)
    if ( (n_time_total/tstep_fract) % n_time_chunks != 0):
        print("Please make sure the total number of time-steps %d from the coarse solver is divisible by the number of Parareal time chunks %d, aborting..." % (n_time_total/tstep_fract,n_time_chunks) )
        sys.exit(0)
    n_tsteps_fine   = int(n_time_total / n_time_chunks) # number of time steps per each Parareal chunk of fine solver
    n_tsteps_coarse = int(n_time_total / tstep_fract / n_time_chunks) # number of time steps per each Parareal chunk of coarse solver
    print("Number of time-steps in each fine   parareal chunk: %d" % (n_tsteps_fine) )
    print("Number of time-steps in each coarse parareal chunk: %d" % (n_tsteps_coarse) )
    if (n_tsteps_fine % nout_fine != 0):
        print("Please make sure the number of time-steps %d is divisible by nout=%d in the fine solver, aborting..." % (n_tsteps_fine,nout_fine))
        sys.exit(0)
    if (n_tsteps_coarse % nout_coarse != 0):
        print("Please make sure the number of time-steps %d is divisible by nout=%d in the coarse solver, aborting..." % (n_tsteps_coarse,nout_coarse))
        sys.exit(0)
    wait_coarse     = 10 # number of seconds to wait for a coarse run to finish (can be faster for debugs with low-resolution cases)
    wait_fine       = 10 # number of seconds to wait for a fine run to finish (can be faster for debugs with low-resolution cases)

    # --- Class to generate directory names consistently
    DN = DirectoryNames(cwd)

    # --- Check that all necessary items are in place before starting
    dirs_to_check = (coarse_solver_dir, fine_solver_dir, io_tools_dir)
    for dir_tmp in dirs_to_check:
        if not os.path.exists(dir_tmp):
            print('You are missing the "'+dir_tmp+'" directory, aborting...') ; sys.stdout.flush()
            sys.exit(1)

    dirs_to_check  = (coarse_solver_dir, fine_solver_dir)
    files_to_check = ("input", "jorek_exec")
    for dir_tmp in dirs_to_check:
        for file_tmp in files_to_check:
            if not os.path.exists(os.path.join(dir_tmp,file_tmp)):
                print('You are missing the file "%s" from the directory "%s", aborting...' % (file_tmp,dir_tmp) ) ; sys.stdout.flush()
                sys.exit(1)

    files_to_check = ("jorek2_parareal_project", "jorek2_extract_data", "jorek2_extract_gauss_points", "jorek2_extract_gauss_data", "jorek2_extract_on_grid",
                      "submit_job.sh", "slurm_script_coarse.sh", "slurm_script_fine.sh", "slurm_script_reference.sh",
                      "my_change_file.perl", "interp_grid.py", "predictor_corrector.py", "run_live_extraction.py")
    for file_tmp in files_to_check:
        if not os.path.exists(os.path.join(io_tools_dir,file_tmp)):
            print('You are missing the file "%s" from the "io_tools" directory, aborting...' % (file_tmp) ) ; sys.stdout.flush()
            sys.exit(1)

    # --- If the coarse solver is not JOREK itself, then there needs to be a coarse_grid.h5 file
    if (coarse_not_jorek):
        if not os.path.exists(os.path.join(coarse_solver_dir,"coarse_grid.h5")):
            print('You are missing the file "%s" from the directory "%s"' % ("coarse_grid.h5",coarse_solver_dir) ) ; sys.stdout.flush()
            print('Running with a non-JOREK coarse solver requires an input file that includes the spatial grid') ; sys.stdout.flush()
            print('on which your coarse solver is running. This is essential to allow the interpolation of data') ; sys.stdout.flush()
            print('between the coarse and fine solvers for the predictor/corrector. Aborting...') ; sys.stdout.flush()
            sys.exit(1)
        if not os.path.exists(os.path.join(coarse_solver_dir,"coarse_solver.py")):
            print('You are missing the file "%s" from the directory "%s"' % ("coarse_solver.py",coarse_solver_dir) ) ; sys.stdout.flush()
            print('Running with a non-JOREK coarse solver requires an coarse_solver.py wrapper to your coarse solver. Aborting...') ; sys.stdout.flush()
            sys.exit(1)
        # --- Note the heavy NN-model file should be stored separately, otherwise if inside "coarse_solver_dir" it will get copied into each "coarse_run_dir", which is very slow
        if not os.path.exists(os.path.join("./","NN_model.pt")):
            print('You are missing the file "%s" from the directory "%s"' % ("NN_model.pt","./") ) ; sys.stdout.flush()
            print('Running with a non-JOREK coarse solver assumes you require a NN-model file. Aborting...') ; sys.stdout.flush()
            sys.exit(1)

    # --- Easy wrapper for file modifications (based on an old perl script)
    CF = ChangeFile( CMD, os.path.join(cwd,io_tools_dir) )

    # --- Directory to save grid reference files that will be the same for all runs
    CMD.exec("mkdir -p grid_references")



    # -----------------------------
    # --- Start Parareal run ------
    # -----------------------------

    # --- Send a job with the full simulation for reference
    if not cli.args.no_reference_run:
        if not os.path.exists("./run_fine_target"): os.mkdir("./run_fine_target")
        CMD.exec("cp "+os.path.join(fine_solver_dir,"*")+" ./run_fine_target/")
        CMD.exec("cp "+os.path.join(io_tools_dir,"slurm_script_reference.sh")+" ./run_fine_target/slurm_script.sh")
        CF.change_file("./run_fine_target/input", "restart", "restart = .f. !")
        CF.change_file("./run_fine_target/input", "nstep", "nstep = "+str(n_time_total)+" !")
        CF.change_file("./run_fine_target/slurm_script.sh", "EXECUTABLE_PLACEHOLDER", "mpirun ./jorek_exec < input > output.txt")
        CMD.exec("cd ./run_fine_target/ ; "+os.path.join(cwd,io_tools_dir,"submit_job.sh")+" slurm_script.sh ; cd ..")
        print("Reference fine-solver run submitted...") ; sys.stdout.flush()

    # --- Parareal iterations
    for i_para in range(n_iter_max):
        iter_dir = DN.para_dir_path(i_para)

        # --- Are we using restart with existing runs?
        if (i_para < restart_para):
            if not os.path.exists(iter_dir):
                print("Warning: you are trying to restart Parareal at cycle #%d, \
                       but the directory \n  %s\ndoes not exist! Aborting..." \
                       % (restart_para,iter_dir) \
                     )
                sys.stdout.flush()
                sys.exit(0)
            continue
        if not os.path.exists(iter_dir): os.mkdir(iter_dir)

        # --- Loop over each time chunk
        for i_chunk in range(i_para,n_time_chunks):
            coarse_dir = DN.coarse_dir_path(i_para,i_chunk)
            fine_dir   = DN.fine_dir_path(i_para,i_chunk)

            # --- Are we using restart with existing runs?
            if ( (i_para <= restart_para) and (i_chunk < restart_chunk) ):
                for dir_tmp in (coarse_dir, fine_dir):
                    if not os.path.exists(dir_tmp):
                        print("Warning: you are trying to restart Parareal at cycle #%d and time-chunk #%d, \
                               but the directory \n  %s\ndoes not exist! Aborting..." \
                               % (restart_para,restart_chunk,dir_tmp) \
                             )
                        sys.stdout.flush()
                        sys.exit(0)
                continue
            if not os.path.exists(coarse_dir): os.mkdir(coarse_dir)
            if not os.path.exists(fine_dir):   os.mkdir(fine_dir)

            # --- Prepare run directories
            CMD.exec( "cp -a "+os.path.join(coarse_solver_dir,"*")+" "+coarse_dir )
            CMD.exec( "cp -a "+os.path.join(fine_solver_dir,  "*")+" "+fine_dir )
            CMD.exec( "cp "+os.path.join(io_tools_dir,"slurm_script_coarse.sh")+" "+os.path.join(coarse_dir,"slurm_script.sh") )
            CMD.exec( "cp "+os.path.join(io_tools_dir,"slurm_script_fine.sh")  +" "+os.path.join(fine_dir,"slurm_script.sh") )

            # -------------------------------------
            # --- Case-0: the very first run ------
            # -------------------------------------
            # --- If this is the very first run, we get the grids to define the gauss points, which will be needed for all later runs
            if ( (i_para == 0) and (i_chunk == 0) ):
                for dir_tmp in (coarse_dir, fine_dir):
                    if checkpoint_rst:
                        if (dir_tmp == fine_dir): 
                            CMD.exec("cd "+dir_tmp+" ; cp "+os.path.join(cwd,checkpoint_dir,"jorek00000.h5")+" . ; cd "+cwd)
                            continue
                        else:
                            if (coarse_not_jorek): continue
                    # --- Make sure the input files and slurm scripts have the correct entries
                    CF.change_file(os.path.join(dir_tmp,"input"), "restart", "restart = .f. !")
                    CF.change_file(os.path.join(dir_tmp,"input"), "nstep", "nstep = 0 !") # zero tsteps because we just want the grid
                    CF.change_file(os.path.join(dir_tmp,"slurm_script.sh"), "EXECUTABLE_PLACEHOLDER", "mpirun ./jorek_exec < input > output.txt")
                    # --- Submit run
                    if (dir_tmp == coarse_dir) and coarse_not_slurm:
                        CMD.exec("cd "+dir_tmp+" ; mpirun ./jorek_exec < input > output.txt ; cd "+cwd)
                    else:
                        CMD.exec("cd "+dir_tmp+" ; "+os.path.join(cwd,io_tools_dir,"submit_job.sh")+" slurm_script.sh ; cd "+cwd)
                print("First grid runs submitted") ; sys.stdout.flush()
                # --- Once the job is submitted, we need to wait for the output to appear
                expected_file = "jorek00000.h5"
                expected_file1 = os.path.join(coarse_dir,expected_file)
                expected_file2 = os.path.join(fine_dir,  expected_file)
                print("Awaing file %s..." % (expected_file) ) ; sys.stdout.flush()
                count_minutes = 0
                while True :
                    if ( os.path.exists(expected_file1) and os.path.exists(expected_file2) ):
                        print("Initial grid runs finished") ; sys.stdout.flush()
                        break
                    else:
                        print("Waiting for initial grid runs to finish... Total elapsed time: %d sec" % (count_minutes*wait_coarse) ) ; sys.stdout.flush()
                    if ( checkpoint_rst and coarse_not_jorek and os.path.exists(expected_file2) ):
                        CMD.exec( "cp "+os.path.join(fine_dir,"*")+" "+coarse_dir )
                        CMD.exec( "cp "+os.path.join(io_tools_dir,"slurm_script_coarse.sh")+" "+os.path.join(coarse_dir,"slurm_script.sh") )
                        continue
                    count_minutes += 1
                    time.sleep(wait_coarse)
                # --- Get Gaussian points from each run
                for dir_tmp in (coarse_dir, fine_dir):
                    extract_gauss_cmd = os.path.join(cwd,io_tools_dir,"jorek2_extract_gauss_points")+" < input -i jorek00000.h5 -o gauss_points.h5"
                    CMD.exec("cd "+dir_tmp+"; "+extract_gauss_cmd+" ; cd "+cwd)
                CMD.exec("cp "+os.path.join(fine_dir,  "gauss_points.h5")+" grid_references/gauss_points_fine.h5")
                CMD.exec("cp "+os.path.join(coarse_dir,"gauss_points.h5")+" grid_references/gauss_points_coarse.h5")
                safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_points_fine.h5"),  os.path.join(fine_solver_dir,  "gauss_points.h5"))
                safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_points_coarse.h5"), os.path.join(coarse_solver_dir,"gauss_points.h5"))
                # --- Create a reference Gauss data file in case the non-JOREK coarse solver has a smaller grid
                # --- This is because it can lead to undetermined (nan) values when projecting the coarse data back onto the JOREK gauss points of the fine solver
                # --- These undetermined values can simply be replaced the the initial conditions using this reference Gauss file, assuming the undetermined values
                # --- are all very close to the boundary conditions (hence shouldn't change)
                if (coarse_not_jorek):
                    extract_gauss_cmd = os.path.join(cwd,io_tools_dir,"jorek2_extract_gauss_data")+" < input -i jorek00000.h5 -o gauss_data_ref.h5"
                    CMD.exec("cd "+fine_dir+"; "+extract_gauss_cmd+" ; cd "+cwd)
                    CMD.exec("cp "+os.path.join(fine_dir,"gauss_data_ref.h5")+" grid_references/")
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_data_ref.h5"), os.path.join(coarse_dir,       "gauss_data_ref.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_data_ref.h5"), os.path.join(coarse_solver_dir,"gauss_data_ref.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_data_ref.h5"), os.path.join(fine_solver_dir  ,"gauss_data_ref.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_data_ref.h5"), os.path.join(checkpoint_dir   ,"gauss_data_ref.h5"))
                if (coarse_not_jorek):
                    # --- Create fine grid file for coarse solver
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_points_fine.h5"),   os.path.join(checkpoint_dir,   "fine_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_points_fine.h5"),   os.path.join(fine_solver_dir,  "fine_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_points_coarse.h5"), os.path.join(coarse_solver_dir,"fine_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_points_fine.h5"),   os.path.join(fine_dir,         "fine_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/gauss_points_coarse.h5"), os.path.join(coarse_dir,       "fine_grid.h5"))
                    # --- Create coarse grid file for fine solver
                    # --- Check if the file has st-info in it
                    with h5py.File(os.path.join(coarse_solver_dir,"coarse_grid.h5"), 'r') as h5file:
                        all_keys = list(h5file.keys())
                        if not ('s_mesh(nR,nZ)' in all_keys):
                            if (os.path.islink(os.path.join(coarse_solver_dir,"coarse_grid.h5"))):
                                CMD.exec("cp "+os.path.join(coarse_solver_dir,"coarse_grid.h5")+" grid_references/coarse_grid_RZ_format.h5.tmp")
                                CMD.exec("rm -f grid_references/coarse_grid_RZ_format.h5")
                                CMD.exec("mv grid_references/coarse_grid_RZ_format.h5.tmp grid_references/coarse_grid_RZ_format.h5")
                            else:
                                CMD.exec("mv "+os.path.join(coarse_solver_dir,"coarse_grid.h5")+" grid_references/coarse_grid_RZ_format.h5")
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/coarse_grid_RZ_format.h5"), os.path.join(coarse_solver_dir,"coarse_grid.h5"))
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/coarse_grid_RZ_format.h5"), os.path.join(fine_dir,         "coarse_grid.h5"))
                            extract_gauss_cmd = os.path.join(cwd,io_tools_dir,"jorek2_extract_on_grid")+" < input -i coarse_grid.h5 -j jorek00000.h5 -o dummy.h5 > output_tmp2.txt"
                            CMD.exec("cd "+fine_dir+"; "+extract_gauss_cmd+" ; cd "+cwd)
                            CMD.exec("cp "+os.path.join(fine_dir,"st_mesh.h5")+" grid_references/")
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/st_mesh.h5"), os.path.join(fine_dir       ,"coarse_grid.h5"))
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/st_mesh.h5"), os.path.join(fine_solver_dir,"coarse_grid.h5"))
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/st_mesh.h5"), os.path.join(checkpoint_dir ,"coarse_grid.h5"))
                        else:
                            if (os.path.islink(os.path.join(coarse_solver_dir,"coarse_grid.h5"))):
                                CMD.exec("cp "+os.path.join(coarse_solver_dir,"coarse_grid.h5")+" grid_references/coarse_grid.h5.tmp")
                                CMD.exec("rm -f grid_references/coarse_grid.h5")
                                CMD.exec("mv grid_references/coarse_grid.h5.tmp grid_references/coarse_grid.h5")
                            else:
                                CMD.exec("mv "+os.path.join(coarse_solver_dir,"coarse_grid.h5")+" grid_references/")
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/coarse_grid.h5"), os.path.join(coarse_solver_dir,"coarse_grid.h5"))
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/coarse_grid.h5"), os.path.join(fine_dir       ,"coarse_grid.h5"))
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/coarse_grid.h5"), os.path.join(fine_solver_dir,"coarse_grid.h5"))
                            safe_symlink(CMD, os.path.join(cwd, "grid_references/coarse_grid.h5"), os.path.join(checkpoint_dir ,"coarse_grid.h5"))
                else:
                    # --- Create fine grid file for coarse solver
                    extract_gauss_cmd = os.path.join(cwd,io_tools_dir,"jorek2_extract_gauss_data")+" < input -i jorek00000.h5 -o gauss_data_ref.h5 > output_tmp1.txt"
                    CMD.exec("cd "+fine_dir+"; "+extract_gauss_cmd+" ; cd "+cwd)
                    safe_symlink(CMD, os.path.join(cwd,fine_dir,"gauss_data_ref.h5"), os.path.join(coarse_dir,"gauss_data_ref.h5"))
                    extract_gauss_cmd = os.path.join(cwd,io_tools_dir,"jorek2_extract_on_grid")+" < input -g -i gauss_data_ref.h5 -j jorek00000.h5 -o dummy.h5 > output_tmp1.txt"
                    CMD.exec("cd "+coarse_dir+"; "+extract_gauss_cmd+" ; cd "+cwd)
                    CMD.exec("cp "+os.path.join(coarse_dir,"st_gauss.h5")+" grid_references/st_gauss_fine.h5")
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/st_gauss_fine.h5"), os.path.join(coarse_dir       ,"fine_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/st_gauss_fine.h5"), os.path.join(coarse_solver_dir,"fine_grid.h5"))
                    # --- Create coarse grid file for fine solver
                    extract_gauss_cmd = os.path.join(cwd,io_tools_dir,"jorek2_extract_gauss_data")+" < input -i jorek00000.h5 -o gauss_data_tmp.h5 > output_tmp2.txt"
                    CMD.exec("cd "+coarse_dir+"; "+extract_gauss_cmd+" ; cd "+cwd)
                    CMD.exec("mv "+os.path.join(coarse_dir  ,"gauss_data_tmp.h5")+" "+os.path.join(fine_dir))
                    extract_gauss_cmd = os.path.join(cwd,io_tools_dir,"jorek2_extract_on_grid")+" < input -g -i gauss_data_tmp.h5 -j jorek00000.h5 -o dummy.h5 > output_tmp2.txt"
                    CMD.exec("cd "+fine_dir+"; "+extract_gauss_cmd+" ; cd "+cwd)
                    CMD.exec("cp "+os.path.join(fine_dir,"st_gauss.h5")+" grid_references/st_gauss_coarse.h5")
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/st_gauss_coarse.h5"), os.path.join(fine_dir       ,"coarse_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/st_gauss_coarse.h5"), os.path.join(fine_solver_dir,"coarse_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/st_gauss_coarse.h5"), os.path.join(checkpoint_dir ,"coarse_grid.h5"))
                # --- Save JOREK grid to make sure we can always load it (eg. if you need to project a coarse grid onto a fine grid)
                CMD.exec("cp "+os.path.join(coarse_dir,"jorek00000.h5")+" grid_references/jorek_grid_coarse.h5")
                CMD.exec("cp "+os.path.join(fine_dir  ,"jorek00000.h5")+" grid_references/jorek_grid_fine.h5")
                safe_symlink(CMD, os.path.join(cwd,"grid_references/jorek_grid_coarse.h5"), os.path.join(coarse_solver_dir,"jorek_grid.h5"))
                safe_symlink(CMD, os.path.join(cwd,"grid_references/jorek_grid_fine.h5"),   os.path.join(fine_solver_dir,  "jorek_grid.h5"))
                if (checkpoint_rst):
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/jorek_grid_coarse.h5"), os.path.join(coarse_dir,"jorek_grid.h5"))
                    safe_symlink(CMD, os.path.join(cwd,"grid_references/jorek_grid_fine.h5"),   os.path.join(fine_dir,  "jorek_grid.h5"))

                # --- If restarting from an existing simulation, need to run differently
                if (checkpoint_rst):
                    # --- Let's assume the data hasn't been extracted
                    flags_live_extraction = ""
                    if (coarse_not_jorek):      flags_live_extraction = flags_live_extraction + " -coarse_not_jorek "
                    if (dir_tmp == coarse_dir): flags_live_extraction = flags_live_extraction + " -coarse_input "
                    if (multi_chkpt):           flags_live_extraction = flags_live_extraction + " -multi_chkpt "
                    tmp_solver_command = "python3 "+os.path.join(cwd,io_tools_dir,"run_live_extraction.py") \
                        + " "+flags_live_extraction+" -coarse_output -fine_output -main_dir ../ -np "+str(n_time_chunks)+" > extract_output.txt"
                    print("Live-extraction of checkpoint data...") ; sys.stdout.flush()
                    CMD.exec("cd "+checkpoint_dir+"; "+tmp_solver_command+" ; cd "+cwd)
                    # --- Extract the data into the expected format
                    for dir_tmp in (coarse_dir, fine_dir):
                        chkpt_dir = os.path.join("../../",checkpoint_dir) # relative to run-dir
                        files_list, index_list = get_all_checkpoint_files(CMD,dir_tmp,chkpt_dir,cwd, "jorek0*.h5")
                        previous_file = files_list[-1]
                        symlink = previous_file
                        CMD.exec("cd "+dir_tmp+"; ln -s "+symlink+" jorek_fine.h5 ; cd "+cwd)
                        choices_dict["coarse_run"] = dir_tmp==coarse_dir
                        choices_dict["coarse_input"] = False
                        parareal_predictor(CMD, dir_tmp,chkpt_dir,cwd,io_tools_dir,"jorek_fine.h5", choices_dict)

            # --- Now change the input files to do the full chunk run
            for dir_tmp in (coarse_dir, fine_dir):
                CF.change_file(os.path.join(dir_tmp,"input"), "restart", "restart = .t. !")
                if ((i_para != 0) or (i_chunk != 0) or checkpoint_rst):
                    # --- Executable for the non-JOREK coarse solver (needs to become a user-input eventually) (AND need to not copy NN model into every directory!)
                    if ( (coarse_not_jorek) and (dir_tmp == coarse_dir) ):
                        input_list = "jorek_data"+str(n_tsteps_coarse).zfill(5)+".h5"
                        for i_file in range(n_tsteps_coarse-10,n_tsteps_coarse-10*n_input_nn,-10):
                            file_tmp = "jorek_data"+str(i_file).zfill(5)+".h5"
                            input_list = file_tmp+","+input_list
                        NN_model_path = os.path.join(cwd,"NN_model.pt")
                        safe_symlink(CMD, "jorek_data"+str(n_tsteps_coarse).zfill(5)+".h5",   os.path.join(dir_tmp,"jorek00000.h5"))
                        # --- Keep the command which will be run at the end of the main loop, when all jobs are submitted
                        coarse_solver_command = 'python3 ./coarse_solver.py -nr '+str(n_output_nn)+' -nn '+NN_model_path+' -il "'+input_list+'" > output.txt'
                        coarse_solver_command = coarse_solver_command \
                            + " ; python3 "+os.path.join(cwd,io_tools_dir,"run_live_extraction.py") \
                            + " -coarse_not_jorek -coarse_input -coarse_output -multi_chkpt -fine_output -np "+str(n_time_chunks)+" >> output.txt"
                    # --- Executable for the fine solver (and the coarse solver if it's JOREK itself)
                    else:
                        tmp_solver_command = "mpirun ./jorek_exec < input > output.txt"
                        flags_live_extraction = ""
                        if (coarse_not_jorek):      flags_live_extraction = flags_live_extraction + " -coarse_not_jorek "
                        if (dir_tmp == coarse_dir): flags_live_extraction = flags_live_extraction + " -coarse_input "
                        if (multi_chkpt):           flags_live_extraction = flags_live_extraction + " -multi_chkpt "
                        if ( (dir_tmp == coarse_dir) and coarse_not_slurm ):
                            tmp_solver_command = tmp_solver_command + " ; "
                        else:
                            tmp_solver_command = tmp_solver_command + "\n"
                        tmp_solver_command = tmp_solver_command \
                            + "python3 "+os.path.join(cwd,io_tools_dir,"run_live_extraction.py") \
                            + " "+flags_live_extraction+" -coarse_output -fine_output -np "+str(n_time_chunks)+" > extract_output.txt" 
                        if ( (dir_tmp == coarse_dir) and coarse_not_slurm ):
                            coarse_solver_command = tmp_solver_command
                        else:
                            CF.change_file(os.path.join(dir_tmp,"slurm_script.sh"), "EXECUTABLE_PLACEHOLDER", tmp_solver_command)
            CF.change_file(os.path.join(fine_dir,  "input"), "nstep", "nstep = "+str(n_tsteps_fine)+" !")
            CF.change_file(os.path.join(coarse_dir,"input"), "nstep", "nstep = "+str(n_tsteps_coarse)+" !")

            # --- Link to the previous checkpoint if needed
            if (i_chunk > 0):
                # --- Interestingly, both fine and coarse solvers restart from the same points
                # --- only the projection against their respective grid changes, which is why we 
                # --- saved the "gauss_points.h5" and "jorek_grid.h5" files earlier...
                for dir_tmp in (coarse_dir, fine_dir):
                    # -----------------------------------------------------------------------
                    # --- Case-1: all remaining chunks of the first Parareal iteration ------
                    # -----------------------------------------------------------------------
                    # --- In the first Parareal cycle, there is no predictor-corrector, we just run through the coarse simulation
                    # --- And kick off a fine run at each chunk as well
                    if ( (i_para == 0) and (i_chunk > 0) ):
                        previous_file = "jorek"+str(n_tsteps_coarse).zfill(5)+".h5"
                        symlink, chkpt_dir =  DN.get_symlink_file(i_para,i_chunk-1,"coarse",previous_file)
                        CMD.exec("cd "+dir_tmp+"; ln -s "+symlink+" jorek_coarse.h5 ; cd "+cwd)
                        choices_dict["coarse_run"] = dir_tmp==coarse_dir
                        choices_dict["coarse_input"] = True
                        parareal_predictor(CMD, dir_tmp,chkpt_dir,cwd,io_tools_dir,"jorek_coarse.h5", choices_dict)
                    
                    # --------------------------------------------------------------
                    # --- Case-2: the first chunk inside later Parareal cycles -----
                    # --------------------------------------------------------------
                    # --- For >0 Parareal cycles, there is no predictor-corrector for the first chunk. The restart checkpoint 
                    # --- of the first chunk needs to be the fine solution propagated from initial point,
                    # --- which which we already know to be 100% accurate
                    if (i_chunk == i_para):
                        previous_file = "jorek"+str(n_tsteps_fine).zfill(5)+".h5"
                        symlink, chkpt_dir =  DN.get_symlink_file(i_para-1,i_chunk-1,"fine",previous_file)
                        CMD.exec("cd "+dir_tmp+"; ln -s "+symlink+" jorek_fine.h5 ; cd "+cwd)
                        choices_dict["coarse_run"] = dir_tmp==coarse_dir
                        choices_dict["coarse_input"] = False
                        parareal_predictor(CMD, dir_tmp,chkpt_dir,cwd,io_tools_dir,"jorek_fine.h5", choices_dict)
                    
                    # --------------------------------------------------------------
                    # --- Case-3: all remaining chunk of later Parareal cycles -----
                    # --------------------------------------------------------------
                    # --- For all the other cases, we need to apply the predictor-corrector, which
                    # --- requires 3 files from previous runs
                    if ( (i_chunk > i_para) and (i_para > 0) ):
                        # --- The coarse file from previous chunk
                        previous_file = "jorek"+str(n_tsteps_coarse).zfill(5)+".h5"
                        symlink, chkpt_dir_coarse =  DN.get_symlink_file(i_para,i_chunk-1,"coarse",previous_file)
                        CMD.exec("cd "+dir_tmp+"; ln -s "+symlink+" jorek_coarse.h5 ; cd "+cwd)
                        # --- Coarse file from previous chunk of previous Parareal iteration
                        previous_file = "jorek"+str(n_tsteps_coarse).zfill(5)+".h5"
                        symlink, chkpt_dir_coarse_prev =  DN.get_symlink_file(i_para-1,i_chunk-1,"coarse",previous_file)
                        CMD.exec("cd "+dir_tmp+"; ln -s "+symlink+" jorek_coarse_prev.h5 ; cd "+cwd)
                        # --- Fine file from previous chunk of previous Parareal iteration
                        previous_file = "jorek"+str(n_tsteps_fine).zfill(5)+".h5"
                        symlink, chkpt_dir_fine =  DN.get_symlink_file(i_para-1,i_chunk-1,"fine",previous_file)
                        CMD.exec("cd "+dir_tmp+"; ln -s "+symlink+" jorek_fine.h5 ; cd "+cwd)
                        # --- additional inputs
                        chkpt_dirs = {"chkpt_dir_coarse": chkpt_dir_coarse, 
                                      "chkpt_dir_coarse_prev": chkpt_dir_coarse_prev,
                                      "chkpt_dir_fine": chkpt_dir_fine}
                        choices_dict["coarse_run"] = dir_tmp==coarse_dir
                        # --- predictor-corrector
                        parareal_predictor_corrector(CMD, dir_tmp,chkpt_dirs,cwd,io_tools_dir, choices_dict)
                    
            # --- Submit fine run
            CMD.exec("cd "+fine_dir+" ; "+os.path.join(cwd,io_tools_dir,"submit_job.sh")+" slurm_script.sh ; cd "+cwd)
            print("Fine run #%d submitted" % (i_chunk) ) ; sys.stdout.flush()
            # --- Submit coarse run
            if coarse_not_slurm:
                print("Coarse run #%d submitted interactively" % (i_chunk) ) ; sys.stdout.flush()
                CMD.exec("cd "+coarse_dir+" ; "+coarse_solver_command+" ; cd "+cwd)
            else:
                CMD.exec("cd "+coarse_dir+" ; "+os.path.join(cwd,io_tools_dir,"submit_job.sh")+" slurm_script.sh ; cd "+cwd)
                print("Coarse run #%d submitted" % (i_chunk) ) ; sys.stdout.flush()
            # --- Once the job is submitted, we need to wait for the output to appear
            expected_file = "jorek"+str(n_tsteps_coarse).zfill(5)+".h5"
            expected_file1 = os.path.join(coarse_dir,expected_file)
            expected_file_extract = "jorek_extracted_fine"+str(n_tsteps_fine).zfill(5)+".h5"
            expected_file2 = os.path.join(coarse_dir,expected_file_extract)
            print("Awaing files %s and %s..." %(expected_file,expected_file_extract) ) ; sys.stdout.flush()
            count_minutes = 0
            while True :
                if os.path.exists(expected_file1) and os.path.exists(expected_file2):
                    print("Coarse run #%d finished" % (i_chunk) ) ; sys.stdout.flush()
                    break
                else:
                    print("Waiting for coarse run #%d to finish... Total elapsed time: %d sec" % (i_chunk,count_minutes*wait_coarse) ) ; sys.stdout.flush()
                count_minutes += 1
                time.sleep(wait_coarse)

        # --- Once all jobs of a Parareal iteration are submitted, we need to wait for the output to appear in all fine-run time-chunk directories
        for i_chunk in range(i_para,n_time_chunks):
            expected_file = "jorek_extracted_fine"+str(n_tsteps_fine).zfill(5)+".h5"
            fine_dir   = DN.fine_dir_path  (i_para,i_chunk)
            expected_file1 = os.path.join(fine_dir,expected_file)
            coarse_dir = DN.coarse_dir_path(i_para,i_chunk)
            expected_file2 = os.path.join(fine_dir,expected_file)
            print("Awaing file %s..." %(expected_file) ) ; sys.stdout.flush()
            count_minutes = 0
            while True :
                if os.path.exists(expected_file1) and os.path.exists(expected_file2):
                    print("Fine and Coarse runs #%d finished" % (i_chunk) ) ; sys.stdout.flush()
                    break
                else:
                    print("Waiting for runs #%d to finish... Total elapsed time: %d sec" % (i_chunk,count_minutes*wait_fine) ) ; sys.stdout.flush()
                count_minutes += 1
                time.sleep(wait_fine)

        



    # --- Finished
    print('Main program finished.') ; sys.stdout.flush()





# --- Main function execution as a script...        
if __name__ == '__main__':
    main()

