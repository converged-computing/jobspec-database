"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as mpl
from mpi4py import MPI

#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Create filename based on current date and time.
        current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
        filename = "LL-Output-{:s}.txt".format(current_datetime)
        FileOut = open(filename,"w")
        # Write a header with run parameters

        print("#=====================================================",file=FileOut)
        print("# File created:        {:s}".format(current_datetime),file=FileOut)
        print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
        print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
        print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
        print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
        print(f"# Number of Cores      {size}",file=FileOut)
        print("#=====================================================",file=FileOut)
        print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
        print("#=====================================================",file=FileOut)
        # Write the columns of data
        for i in range(nsteps+1):
            print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
        FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
def all_energy(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
def get_order(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()
#=======================================================================
def MC_step(arr,Ts,nmax,comm, rank, size):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
      comm (MPI comm object) = MPI communicator.
      rank (int) = the cpu rank
      size (int) = the number of mpi threads
    Description:
      mpi version of MC_step
    Returns:
      accept/(local_nmax**2) (float) = acceptance ratio for current MCS.
    """
    
    #calculates local values for nmax depending on the size
    local_nmax = nmax // size
    
    #calculates the start and end points of the region of the lattice that will be computed by this mpi thread
    start = rank * local_nmax
    if rank != size - 1:
        end = (rank + 1) * local_nmax
    else:
        end = nmax

    #initilises an acceptance ratio of 0
    local_accept = 0
    
    #initilizes the arrays for xran and yran
    xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    yran = np.random.randint(0,high=nmax, size=(nmax,nmax))

    #loops over a slice of the lattice
    for i in range(start, end):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = np.random.normal(scale=0.1+Ts)
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                local_accept += 1
            else:
                boltz = np.exp( -(en1 - en0) / Ts )
                if boltz >= np.random.uniform(0.0,1.0):
                    local_accept += 1
                else:
                    arr[ix,iy] -= ang

    # do a reduction to bring together the acceptance ratios accross the processes
    total_accept = comm.reduce(local_accept, op=MPI.SUM, root=0)

    # rank 0 returns the acceptance ratio
    if rank == 0:
        return total_accept / (nmax**2)
    else:
        return None

#=======================================================================

def main(program, nsteps, nmax, temp, pflag, comm, rank, size):
    """
    Arguments:
        program (string) = the name of the program;
        nsteps (int) = number of Monte Carlo steps (MCS) to perform;
        nmax (int) = side length of square lattice to simulate;
        temp (float) = reduced temperature (range 0 to 2);
        pflag (int) = a flag to control plotting.
        comm (MPI communicator) = MPI communicator.
    Description:
        This is the main function running the Lebwohl-Lasher simulation.
    Returns:
        NULL
    """
    # Create and initialise lattice
    lattice = initdat(nmax)

    if rank == 0:
        # Plot initial frame of lattice
        #plotdat(lattice,pflag,nmax)
        # Create arrays to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps+1,dtype=np.dtype)
        ratio = np.zeros(nsteps+1,dtype=np.dtype)
        order = np.zeros(nsteps+1,dtype=np.dtype)
        # Set initial values in arrays
        energy[0] = all_energy(lattice,nmax)
        ratio[0] = 0.5 # ideal value
        order[0] = get_order(lattice,nmax)
        # Begin doing and timing some MC steps.
        initial = time.time()

    for it in range(1,nsteps+1):
        accept_ratio = MC_step(lattice,temp,nmax,comm, rank, size)
        if rank == 0:
            ratio[it] = accept_ratio
            energy[it] = all_energy(lattice,nmax)
            order[it] = get_order(lattice,nmax)

    if rank == 0:
        final = time.time()
        runtime = final-initial
        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        #plotdat(lattice,pflag,nmax)



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, comm, rank, size)
    else:
        print("Usage: mpiexec -n <num_processes> python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================