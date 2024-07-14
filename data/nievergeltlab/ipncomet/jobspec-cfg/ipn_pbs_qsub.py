import os, re, sys
import subprocess
import time
import types
import socket
import string
import ipn_util

from random import choice

# This is a whole set of callable functions, the qsub script will generate 
# a file ended with error_suffix ('ERROR'). This indicates that there
# is an error in the running script. By default, there is a hidden directory
# called '.tmp' in the directory issuing qsub command (thus calling run_XXX 
# serial of functions).

__error_suffix__    = 'ERROR'
__qsub_err_suffix__ = 'qerr'
__qsub_out_suffix__ = 'qout'
__qsub_sh_suffix__  = 'qsub'

__my_qsub_sleep_time__ = 120	# polling time every 5 minutes

svr_port = 0			# DO NOT USE daemon style
succeed_str='SUCCEED'		# indicates a job succeeds
fail_str   ='FAIL'		# indicates a job fails

qsub_tmp = '.tmp'		# store ouput/input files
qsub_dir = '.qsub'		# store qsub shell files

signal_command = 'qjob_done'
begin_with_num = re.compile ( '^[0-9X]' )
MIN_NUM_JOBS=200 
HEAVY_JOBS_N=30
__WHERE__=sys.stderr

# when the module is loaded, check if there are required directories under
# the current directory
if ( not os.path.exists(qsub_tmp) ): 
	ipn_util.check_or_make_dir (qsub_tmp)

if ( not os.path.exists(qsub_dir) ): 
	ipn_util.check_or_make_dir (qsub_dir)


# A class which wraps the individual procedure, mainly for 
# choosing the port
class QJobs:
    def __init__ ( self, port = 0 ):
	self.port = port
	self.jobs = [ ]

    def run ( self, commands ):
	self.jobs = [ ]

	for nm in commands.keys():
	    jname = write_qsub_file ( nm, commands[nm], 8, self.port )
	    self.jobs.append ( jname )
	    print jname

	if ( self.port == 0 ): run_wait_poll (self.jobs)
	else: run_wait_daemon ( self.jobs, self.port )

	cleanup ( self.jobs )
# ------------------------------------------------------------------------|


class QsubError (Exception):
    def __init__ ( self, value ):
	self.value = value
    def __str__ (self):
	return repr (self.value)
# ----------------------------------------------------------------------- |


class QsubSignal:
    def __init__ (self, sigfile, jobid, ajob):
	self.sigfile = sigfile
	self.jobid   = jobid
	self.qsubfile= ajob
# ----------------------------------------------------------------------- |


def randstr (length=8, chars=string.letters + string.digits):
    return ''.join([choice(chars) for i in range(length)])
# -------------------------------------------------------------------------|


# submit a job and return the submitted job's jobid
def myqsub (shfile, q=None):
    qsubpipe = None

    try:
	if q is None:
	    qsubpipe = subprocess.Popen (['qsub', shfile], 
					  shell=False,
					  stdout=subprocess.PIPE, 
					  stderr=subprocess.PIPE)
	else:
	    qsubpipe = subprocess.Popen (['qsub', '-P', q, shfile],
					  shell=False,
					  stdout=subprocess.PIPE, 
					  stderr=subprocess.PIPE)
    except OSError, detail:
	raise detail
    except ValueError, e:
	raise e

    stdout_data, stderr_data = qsubpipe.communicate()

    E = stdout_data.rstrip('\n').split(' ')

    if ( stderr_data is not None and stderr_data != '' ): 
	raise QsubError (stderr_data)

    return stdout_data
# --------------------------------------------------------------------- |


# Changed just by returning just days/hours/minutes/seconds
def get_time ( seconds ):
    '''Convert given seconds into days:hours:seconds.'''
    result = ':'

    if ( seconds > 24*60*60 ): 
	return str(seconds/(24*60*60.0)) + ' ' + 'days'

    if ( seconds > 60*60 ):
	return str(seconds/(60*60.0)) + ' ' + 'hours'

    if ( seconds > 60 ):
	return str(seconds/60.0) + ' ' + 'minutes'

    return str(seconds) + ' seconds'
#------------------------------------------------------------------------|


def get_qsub_sh_name ( key, dirname=qsub_dir ):
    #return os.path.join ( key, __qsub_sh_suffix__ )
    return os.path.join ( dirname, '.'.join ( [key, __qsub_sh_suffix__] ) )
# -------------------------------------------------------------------------|


def get_qsub_qerr_name ( key, tmp_dir=qsub_tmp ):
    return os.path.join (tmp_dir, '.'.join ([key, __qsub_err_suffix__]))
# -------------------------------------------------------------------------|


def get_qsub_qout_name ( key, tmp_dir=qsub_tmp ):
    return os.path.join (tmp_dir, '.'.join ([key, __qsub_out_suffix__]))
# -------------------------------------------------------------------------|


def get_qsub_error_name ( key, tmp_dir=qsub_tmp ):
    return os.path.join (tmp_dir, '.'.join ([key, __error_suffix__]))
# -------------------------------------------------------------------------|


def get_qsub_signal_name ( key, tmp_dir=qsub_tmp ):
    key = get_key_from_qsub_names(key)

    # format: .tmp/.key.qsub
    return os.path.join (tmp_dir, '.'.join (['', key, __qsub_sh_suffix__]) )
# -------------------------------------------------------------------------|


def get_key_from_qsub_names ( qsub_name ):
    a = os.path.basename (qsub_name)
    return (re.sub ( '|'.join (['\.'+__error_suffix__+'$',
				'\.'+__qsub_err_suffix__+'$',
				'\.'+__qsub_out_suffix__+'$',
				'\.'+__qsub_sh_suffix__+'$']), '', a) )
# -------------------------------------------------------------------------|

def write_qsub_file ( set_name, exe_cmd ):
    return __write_qsub_file__ (set_name, exe_cmd, 2, '\n', '\n')

def write_2G_qsub_file ( set_name, exe_cmd ):
    return __write_qsub_file__ (set_name, exe_cmd, 2)
# -------------------------------------------------------------------------|


def write_4G_qsub_file ( set_name, exe_cmd ):
    return __write_qsub_file__ (set_name, exe_cmd, 4)
# -------------------------------------------------------------------------|


def write_8G_qsub_file ( set_name, exe_cmd ):
    return __write_qsub_file__ (set_name, exe_cmd, 8)
# -------------------------------------------------------------------------|


def write_16G_qsub_file ( set_name, exe_cmd ):
    return __write_qsub_file__ (set_name, exe_cmd, 16)
# -------------------------------------------------------------------------|

def write_32G_qsub_file ( set_name, exe_cmd ):
    return __write_qsub_file__ (set_name, exe_cmd, 32)
# -------------------------------------------------------------------------|


def write_qsub_file_multi ( set_name, exe_cmd ):
    return __write_qsub_file_multi_commands__ (set_name, exe_cmd, 2, '\n', '\n')

def write_2G_qsub_file_multi ( set_name, exe_cmd ):
    return __write_qsub_file_multi_commands__ (set_name, exe_cmd, 2)
# -------------------------------------------------------------------------|

def write_4G_qsub_file_multi ( set_name, exe_cmd ):
    return __write_qsub_file_multi_commands__ (set_name, exe_cmd, 4)
# -------------------------------------------------------------------------|

def write_8G_qsub_file_multi ( set_name, exe_cmd ):
    return __write_qsub_file_multi_commands__ (set_name, exe_cmd, 8)
# -------------------------------------------------------------------------|

def write_16G_qsub_file_multi ( set_name, exe_cmd ):
    return __write_qsub_file_multi_commands__ (set_name, exe_cmd, 16)
# -------------------------------------------------------------------------|

def write_32G_qsub_file_multi ( set_name, exe_cmd ):
    return __write_qsub_file_multi_commands__ (set_name, exe_cmd, 32)
# -------------------------------------------------------------------------|


# The real underline function
def __write_qsub_file__ (set_name, exe_cmd, mem=2, 
			before_exe='\n', after_exe='\n', 
				port=svr_port, tmp_dir = '.tmp'):
    '''Given the set name, and command, we write a qsub file at 
       the current directory write your qsub files,hope this generalized 
       everything and specify some garbage collection directory'''
    from socket import gethostname

    hostnm = gethostname()

    if ( begin_with_num.match ( set_name ) ): set_name = 'X'+set_name

    set_name = set_name + '.' + randstr()

    qfname = get_qsub_sh_name ( set_name )

    qerr_file = get_qsub_qerr_name  (set_name)
    qout_file = get_qsub_qout_name  (set_name)
    error_file= get_qsub_error_name (set_name)

    # remove old files if there are any
    if ( os.path.exists (qerr_file) ): os.unlink ( qerr_file )
    if ( os.path.exists (qout_file) ): os.unlink ( qout_file )
    if ( os.path.exists (error_file)): os.unlink ( error_file )

    cur_dir = os.getcwd ( )

    # write out qsub file, use tcsh
    f = open( qfname, 'w' )
    print >> f, '#! /bin/bash'
    print >> f, '#PBS -N '+set_name
    print >> f, '#PBS -e '+qerr_file
    print >> f, '#PBS -o '+qout_file
    print >> f, '#PBS -l vmem='+str(mem)+'gb'
    print >> f, '#PBS -l walltime=2:00:00'

    print >> f

    print >> f, 'hostname'
    print >> f

    print >> f, 'source ~/.bashrc'
    print >> f

    print >> f, 'cd ' +  cur_dir
    print >> f

    print >> f, before_exe
    print >> f

    print >> f, exe_cmd
    print >> f

    print >> f, 'flag=$?'
    print >> f

    print >> f, '# If there is something wrong, we generate an error file'
    print >> f, 'if [ $flag -ne 0 ];'
    print >> f, 'then'
    print >> f, '	echo "%s" FAILED > %s' % (exe_cmd, error_file)
    print >> f, 'fi'

    print >> f, after_exe
    print >> f

    print >> f, "#Generate a file name indicating job finished"
    print >> f, 'touch ' + get_qsub_signal_name (set_name)

    f.close ( )

    return qfname
# -------------------------------------------------------------------------|


# The commands are a set of sequential commands
# after each command we add a simple test
def __write_qsub_file_multi_commands__ (set_name,
					commands,
					mem=2,
					before_exe='\n',
					after_exe='\n',
					port=svr_port,
					tmp_dir = '.tmp'):
    '''Given the set name, and command, we write a qsub file at 
       the current directory write your qsub files,hope this generalized 
       everything and specify some garbage collection directory'''
    from socket import gethostname

    hostnm = gethostname()

    if ( begin_with_num.match ( set_name ) ): set_name = 'X'+set_name

    set_name = set_name + '.' + randstr()

    qfname = get_qsub_sh_name ( set_name )

    qerr_file = get_qsub_qerr_name  (set_name)
    qout_file = get_qsub_qout_name  (set_name)
    error_file= get_qsub_error_name (set_name)

    # remove old files if there are any
    if ( os.path.exists (qerr_file) ): os.unlink ( qerr_file )
    if ( os.path.exists (qout_file) ): os.unlink ( qout_file )
    if ( os.path.exists (error_file)): os.unlink ( error_file )

    if type(commands).__name__ != 'list' and \
		    type(commands).__name__ != 'str':
	raise Exception ('In "__write_qsub_file_multi_command__"'+\
				' we need list or str for commands')
    elif type(commands).__name__ == 'str':
	commands = [commands]

    cur_dir = os.getcwd ( )

    # write out qsub file, use tcsh
    f = open( qfname, 'w' )
    print >> f, '#! /usr/bin/tcsh'
    print >> f, '#$ -N z'+set_name
    print >> f, '#$ -cwd'
    print >> f, '#$ -e '+qerr_file
    print >> f, '#$ -o '+qout_file
    print >> f, '#$ -l h_vmem='+str(mem)+'G'
    print >> f

    print >> f, 'hostname'
    print >> f

    print >> f, 'cd ' +  cur_dir
    print >> f

    print >> f, before_exe
    print >> f

    for exe_cmd in commands:
	print >> f, exe_cmd
	print >> f

	print >> f, 'set flag=$?'
	print >> f

	print >> f, '# If there is something wrong, we generate an error file'
	print >> f, 'if ( $flag != 0 ) then'
	print >> f, '	echo "%s" FAILED > %s' % (exe_cmd, error_file)
	print >> f, '	exit 1'
	print >> f, 'endif'
	print >> f

    print >> f, after_exe
    print >> f

    # signale the processing has been finished, the program try to connect
    # to the given port, if not succeed, it generates a signal file
    print >> f, ' '.join ( [signal_command, '-m', qfname,
					    '-s', hostnm,
					    '-p', str(port),
					    '-r', '$flag' ] )

    f.close ( )

    return qfname
# -------------------------------------------------------------------------|


def run_qsub_cmd ( cmd, hint = '', waiting_time=__my_qsub_sleep_time__, 
					tmp_path = qsub_tmp, qname=None ):
    ''' Given a command, we issue it by using qsub, hint is used to construct
        qsub names.'''

    qfile = write_8G_qsub_file (hint, cmd)

    qfile_list = [qfile]

    name=qname
    ret=run_wait_poll (qfile_list, waiting_time, qname=name)

    if ret is not None: return ret
    else: 
	return cleanup(qfile_list)

    return None
# ----------------------------------------------------------------------- |


def check_qsub_error ( log_path = qsub_tmp ):
    '''Check if there is any qsub file error, 
       if so the set name is returned.'''
    from mysys import myglob

    error_files = myglob (log_path, __error_suffix__+'$')

    return map (lambda nm: re.sub ('\.'+__error_suffix__+'$', '', nm), error_files)
# ----------------------------------------------------------------------- |

def clean_dir ( log_path = qsub_tmp ):
    '''decide if to remove all the files under .tmp'''
    files = os.listdir ( log_path )

    # first round, determine if we need to remove files
    for nm in files:
	# search for error file, if there is one, stop
	if ( re.search ( __error_suffix__, nm ) ): 
	    nm_list = check_qsub_error ( )
	    for err_nm in nm_list:
		print >> __WHERE__, err_nm
	    return

    # second round, it is time to remove all the files
    for nm in files:
	fname = os.path.join ( log_path, nm ) 
	os.remove ( fname )
# ----------------------------------------------------------------------- |


def set_qsub_wait_time ( wait_time ):
    __my_qsub_sleep_time__ = wait_time
# -------------------------------------------------------------------- |


# TESTED
def cleanup ( filelist, rm_qlogs=True ):
    '''Given a list of "qsub" file, delete all of them, at the same time,
       we also delete all the qerr/qout/err files, if there is no error.
    '''

    time.sleep (120)	# wait some time for the files be ready

    errlist = []

    for qsubfile in filelist:
        if ( os.path.exists (qsubfile) ): os.unlink ( qsubfile )

        if not rm_qlogs: continue
        
        bname = os.path.basename (qsubfile)
        keyname = re.sub ( '\.'+__qsub_sh_suffix__+'$', '', bname )

        err_file  = get_qsub_error_name (keyname)
        qerr_file = get_qsub_qerr_name  (keyname)
        qout_file = get_qsub_qout_name  (keyname)

	# if there is error file, append qerr_file
        if os.path.exists(err_file): 
            errlist.append (qerr_file)
        else:
	    if os.path.exists(qerr_file): os.unlink (qerr_file)
	    if os.path.exists(qout_file): os.unlink (qout_file)

    if len(errlist) == 0: return None

    return errlist
# ----------------------------------------------------------------------- |

def get_myjobs ( ):
    import xml_qstat

    return xml_qstat.get_myjobs ()

#    qsubpipe = None
#
#    try:
#	# NOTE only this form works.
#        qsubpipe = subprocess.Popen ('qstat -u `whoami`' '',
#					shell=True, stdout=subprocess.PIPE)
#    except OSError, detail:
#        print >> __WHERE__, detail
#        sys.exit(1)
#    except ValueError, e:
#        print >> __WHERE__, e
#        sys.exit(1)
#
#    stdout_data, stderr_data = qsubpipe.communicate()
#
#    if stdout_data == '': return None
#
#    lines = stdout_data.rstrip('\n').split('\n')
#
#    i = 0
#    while True:
#        if re.search ( '-----', lines[i] ): break
#        else: i += 1
#
#    job_list={}
#    i += 1      # ignore '-----' line
#
#    while i < len(lines):
#        E = lines[i].split()
#        id = E[0].split('.')
#        job_list[id[0]]=id[0]
#        i += 1
#
#    return job_list
# ------------------------------------------------------------------------- |

def get_job_id ( msg ):
    # message from the qsub
    #Your job 1340664 ("STDIN") has been submitted
    #return msg.split()[2]
    return msg.rstrip('\n')
# ------------------------------------------------------------------------- |

# check if there are current running jobs in the waiting jobs
def has_jobs_running (waiting_jobs, current_jobs):
    if (len(waiting_jobs) != 0) and (current_jobs is None): return False

    flag = False

    for a in waiting_jobs:
	    if current_jobs.has_key(a): return True

    return flag
# ------------------------------------------------------------------------- |

# Given a set of job files, which are qsub files,  stored in job_files.
# This procedure submits all of the jobs and wait for them to finish.
# If there are some jobs failed, this function returns the set of
# qsub files
def run_wait_poll ( job_files, sleep_time=__my_qsub_sleep_time__, 
					tmp_dir = qsub_tmp, qname=None ):
    '''Given a set of job files, we check the job process every sleep_time'''
    ipn_util.check_or_make_dir (tmp_dir)

    waiting_files = { }	# record signals

    # submit the jobs part
    for ajob in job_files:
        retmsg = myqsub (ajob, qname)
	id = get_job_id (retmsg)

	sigfile = get_qsub_signal_name (ajob)

	# remove old one
        if ( os.path.exists (sigfile) ): os.unlink ( sigfile )

        #waiting_files[id] = QsubSignal (sigfile, id, ajob)
        waiting_files[id] = QsubSignal (sigfile, None, ajob)

    passed_time = 0

    print >> __WHERE__, 'Polling daemon starts @', \
		    		time.strftime("%a %d %b %Y %H:%M:%S")

    # waiting for the jobs to be done
    while True:
	print >> __WHERE__, 'Number of jobs in the queue: %d' %\
						(len(waiting_files))
	print >> __WHERE__, 'Wait for %d seconds.' % (sleep_time)

        time.sleep ( sleep_time )

        current_jobs = get_myjobs()

	for id in waiting_files.keys():
            if ( os.path.exists ( waiting_files[id].sigfile ) ):
                print >> __WHERE__, 'Found signal file', waiting_files[id].sigfile

                # remove both the file and the key
                os.unlink ( waiting_files[id].sigfile )
                del waiting_files[id]

        passed_time += sleep_time

	if ( len(waiting_files.keys()) == 0 ): 
		print >> sys.stderr, 'It says that there are no waiting files'
		break
    # ---

    err_list = None

    # NOTE it is possible that there are jobs in current_jobs,
    # but the jobs actually have finished; this does not matter.
    if (len(waiting_files) >0):
	err_list = [ ]
	for v in waiting_files.values():
	    err_list.append (v.qsubfile)

    return err_list
# -------------------------------------------------------------------------|

def run_cmd_on_node ( cmd, proc_stdout=subprocess.PIPE,
				proc_stderr=subprocess.PIPE, qname=None ):
    '''
    Given a command, we run the command on the node. NOTE, the cmd
    should be a vector [ls, '-l'] or a simple command (ls).
    '''

    cmdpipe=subprocess.Popen (cmd, shell=False, 
				stdout=proc_stdout, stderr=proc_stderr)

    child_stdout, child_stderr= cmdpipe.communicate()

    return cmdpipe.returncode, child_stdout, child_stderr
# -------------------------------------------------------------------------|

# job_files needed to submit
# the directory we need to remove temp files
# how many jobs we want in the queue each time
# sleep time
def run_with_num (job_files, num_in_queue=1000,
		  sleep_time=__my_qsub_sleep_time__, 
		  tmp_dir='.tmp', qname=None):
    '''
       If the number of jobs is too many, we only submit a certain number of
       jobs each time, however, we keep replenish the jobs, to keep the number
       steady.
    '''

    if ( (job_files is None) or len (job_files) == 0 ): return None

    waiting_files = { }

    # initially, there is no job in the queue
    avail_spots = num_in_queue 

    cur = 0  # points to the first job that has not been submitted.
    total = len (job_files)    # number of jobs not submitted 
    finished_jobs = 0          # number of jobs have been finished

    passed_time = 0

    # waiting for the jobs to be done
    while True :
	this_run = 0	# how many jobs we submitted this run

        # submit jobs. keep the number of jobs in the up to avil_spots
        while ( (avail_spots > 0) and (cur < len(job_files)) ):
            ajob = job_files[cur]

	    retmsg = myqsub (ajob, qname)

	    id = get_job_id (retmsg)

	    sigfile = get_qsub_signal_name (ajob)

	    # remove old one : Adam : could be this file..
	    if ( os.path.exists (sigfile) ): os.unlink ( sigfile )

	    waiting_files[id] = QsubSignal (sigfile, id, ajob)

            cur += 1
            avail_spots -= 1
            total -= 1

	    this_run += 1
	# end of while

	print >> __WHERE__, 'number of jobs FINISHED %d' % (finished_jobs)
	print >> __WHERE__, 'number of jobs IN the queue %d' % \
                                        (num_in_queue-avail_spots)
	print >> __WHERE__, 'number of jobs NOT submitted %d' % (total)
	print >> __WHERE__, 'number of jobs THIS time submitted %d'%(this_run)

	print >> __WHERE__, 'sleep time %d' % (sleep_time)

	time.sleep ( sleep_time )

	passed_time += sleep_time

	current_jobs = get_myjobs ()

	print >> sys.stderr, 'current jobs:', current_jobs
	print >> sys.stderr, 'waiting jobs:', waiting_files.keys()

        # record finished jobs
	for akey in waiting_files.keys():

	    if ( os.path.exists ( waiting_files[akey].sigfile ) ):
		print >> __WHERE__, "Found signal file:", \
						waiting_files[akey].sigfile

		# remove indication file from disk
		os.unlink ( waiting_files[akey].sigfile ) 

		del waiting_files[akey] # delete the key from the list

		avail_spots   += 1
		finished_jobs += 1 
        # end of for loop

        print >> __WHERE__, 'Number of jobs LEFT in the queue (after ' + \
					 get_time(passed_time) + ') = ' + \
					 str (len(waiting_files))

	# there are current-running jobs, and no more jobs, get out
 #Adam: I think it is stopping because jobs are not running yet, but this terminates the loop
 #Basically what this is doing is, as long as all jobs have been submitted, it's going to terminate the while loop
 #I make sure that there are no waiting files before it terminates!!
	if not has_jobs_running (waiting_files.keys(), current_jobs) and \
						cur >= len(job_files) and len(waiting_files) == 0: break


    # end of outer while loop

    # wait for 60 seconds, check again, sometimes, network is too slow
    time.sleep (120)

    for akey in waiting_files.keys():
	if ( os.path.exists ( waiting_files[akey].sigfile ) ):
	    print >> __WHERE__, "Found signal file:", \
						waiting_files[akey].sigfile

            # remove indication file from disk
            os.unlink ( waiting_files[akey].sigfile ) 

            del waiting_files[akey] # delete the key from the list

    err_list = None

    if (len(waiting_files) >0):
	err_list = [ ]
	for v in waiting_files.values():
	    err_list.append (v.qsubfile)

    return err_list
# ----------------------------------------------------------------------|

def run_jobs_no_cleanup ( qfiles, is_heavy_io=False, qname=None ):
    inname=qname
    running_results=None

    if is_heavy_io:
	running_results=run_with_num (qfiles, HEAVY_JOBS_N, qname=inname)
    else:
	if len(qfiles) > MIN_NUM_JOBS:
		running_results=run_with_num (qfiles, qname=inname)
	else:
		running_results=run_wait_poll (qfiles, qname=inname)

    if running_results is not None:
	    print >> __WHERE__, 'MISSING JOBS???'
	    print >> __WHERE__, '\t\n'.join (running_results)

    print >> __WHERE__, 'NO CLEANUP'
    print >> __WHERE__, 'NO CLEANUP'
    print >> __WHERE__, 'NO CLEANUP'
# ----------------------------------------------------------------------|


# automatically choose function, and cleanup
def run_jobs ( qfiles, is_heavy_io=False, qname=None ):
    inname=qname

    running_results=None
    if is_heavy_io:
	running_results=run_with_num (qfiles, HEAVY_JOBS_N, qname=inname)
    else:
	if len(qfiles) > MIN_NUM_JOBS:
		running_results = run_with_num (qfiles, qname=inname)
	else:
		running_results = run_wait_poll (qfiles, qname=inname)

    if running_results is not None:
	print >> __WHERE__, 'MISSING JOBS???'
	print >> __WHERE__, '\t\n'.join (running_results)
	return

    ret=cleanup(qfiles)

    if ret is not None:
	print >> __WHERE__, 'THERE MIGHT BE RUNNING ERRORs, CHECK:'
	print >> __WHERE__, '\n'.join (ret)
# ----------------------------------------------------------------------|

# run a series of commands sequentially
def run_commands_seq ( commands ):
    if ( type(commands) != types.DictType and \
					type(commands) != types.ListType):
	raise Exception ('Input should be dictionary/list type',
			 'Input should be dictionary/list type')

    if ( type(commands) == types.DictType ):
	commands = commands.values()

    for cmd in commands:
	print >> __WHERE__, cmd
	os.system (cmd)
# -------------------------------------------------------------------- |

# run a series of commands by parallelly
def run_commands (incommands, qname=None):
    """ Given a sequence of commands, the program decides if we need to 
        run them sequentially or by qsub. """

    commands=incommands
    name=qname
    if ( type(incommands) != types.DictType and \
		    type(incommands) != types.ListType):
	raise Exception ('Input should be dictionary/list type.',
			 'Input should be dictionary/list type.')
    
    # make the commands looke like hash table
    if type(incommands) == types.ListType:
	commands= {}
	i = 0
	while (i < len(incommands)):
		commands[str(i)] = incommands[i]
		i += 1

    qlist = [ ]
    for nm, cmd in commands.iteritems():
	qfile = write_8G_qsub_file (str(os.path.basename(nm)), cmd)
	qlist.append ( qfile )

    run_jobs (qlist, qname=name)
# ----------------------------------------------------------------------- |

def run_command ( command_elements, noqsub=False, name="command_name" ):
    ret=out=err=None

    if noqsub:
        ret, out, err=run_cmd_on_node (command_elements)
    else:
        ret=run_qsub_cmd ( ' '.join (command_elements), name )

    return ret, out, err
# ----------------------------------------------------------------------- |

# for backward compitability
run_and_wait = run_wait_poll
