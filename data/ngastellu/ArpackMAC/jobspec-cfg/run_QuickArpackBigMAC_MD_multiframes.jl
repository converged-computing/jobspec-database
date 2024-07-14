module run_QuickArpackBigMAC
include("./QuickArpackBigMAC.jl")
include("./SpectralLanczos.jl")
include("./CoordsIO.jl")
include("./TightBinding.jl")

using .QuickArpackBigMAC, .SpectralLanczos, .CoordsIO, .TightBinding
using LinearAlgebra, SparseArrays, PyCall, Base.Filesystem

#posfile = expanduser(ARGS[1])
#strucindex = parse(Int,split(split(split(posfile,'/')[end],'-')[2], '_')[1])

temp = 40
frame0_index = 10000
nframes = 3
framestep = 1000

const trajfile::String = expanduser("~/Desktop/simulation_outputs/MAC_MD_lammps/40x40/$(temp)K_norotate_0-100000-500.lammpstrj")
const rCC::Float64 = 1.8 #max nearest neighbour distance in angstrom


println("Reading coords from file: $trajfile...")
flush(stdout)

for n=0:nframes-1

    frame_index = frame0_index + n*framestep
    println("******** Working on frame $frame_index ********")
    flush(stdout)


    fullpos = get_frame(trajfile, frame_index)

    println(size(fullpos))

    py"""import numpy as np
    from qcnico.remove_dangling_carbons import remove_dangling_carbons
    rCC = $rCC
    pos = remove_dangling_carbons($(PyObject(fullpos)),$rCC)
    """

    pos = PyArray(py"pos"o)

    println("Constructing hamiltonian...")
    H = lindbergHtb_sparse(pos,rCC)
    println("Done!")

    py"""import numpy as np
    frame = $frame_index
    np.save(f"H-{frame}.npy",$(PyObject(H)))
    """

    N = size(H,1)

    if N % 2 != 0
        error("Number of atoms needs to be even! We have N = $N.") 
    end

    nhalf = Int(N/2)
    #eps_QCFFPI = 2.7e-7
    eps_tb = 1e-7
    
    print("Estimating eHOMO...")
    approx_eHOMO, Rspectrum = estimate_eHOMO(H,eps_tb*100) #Rspectrum = spectral range
    print("Done! ")
    println("Estimated eHOMO = $(approx_eHOMO) eV")
    println("Spectral range = $Rspectrum eV")

    print("Running one-shot Lanczos... ")
    ε, ψ, iLUMO = LUMO_arpack_MAC(H, approx_eHOMO, Rspectrum)
    nconv = size(ε,1)
    println("Done! Obtained $nconv eigenvalues. iLUMO = $iLUMO")

    py"""import numpy as np
    nn = $frame_index
    ii = $iLUMO
    np.save(f"eARPACK_bigMAC_iLUMO={ii}-{nn}.npy",$(PyObject(ε)))
    np.save(f"MOs_ARPACK_bigMAC_iLUMO={ii}-{nn}.npy",$(PyObject(ψ)))    
    """
end

end