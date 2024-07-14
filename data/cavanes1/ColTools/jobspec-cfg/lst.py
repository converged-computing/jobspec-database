# required files
#   Both endpoint geometries, in COLUMBUS Cartesian format
#   If provided = True, intcfl in the directory this program is being run from

# parameters
step = 0.5 # Step size
provided = False
init_cart_name = "0/geom" # Path of initial Cartesian geometry file
final_cart_name = "geom" # Name of final Cartesian geometry file

# module import
import numpy as np
import os
import subprocess
print("Modules imported")

# read given Cartesian geometry files
f = open(init_cart_name, "r")
init_cart = f.readlines()
f.close()
f = open(final_cart_name, "r")
final_cart = f.readlines()
f.close()
print("Cartesian geometry files read")

# extract data from Cartesian geometries
init_cart_data = []
for line in init_cart:
    init_cart_data.append(line.split())
final_cart_data = []
for line in final_cart:
    final_cart_data.append(line.split())
print("Cartesian geometry data extracted")

# create directories
allitems = os.listdir(".")
if "FINAL" not in allitems:
    os.system("mkdir FINAL")
os.system("cp " + final_cart_name + " FINAL/geom")

if provided:
    os.system("cp intcfl 0")
    os.system("cp intcfl FINAL")
else:
    # generate intcin
    conv = 0.529177211 # Angstroms per Bohr radii
    # initial geometry
    f = open("0/intcin", "w")
    f.write("TEXAS\n")
    for atom in init_cart_data:
        f.write("  " + atom[0]
                + format(float(atom[1]), "17.5f")
                + format(float(atom[2])*conv, "10.5f")
                + format(float(atom[3])*conv, "10.5f")
                + format(float(atom[4])*conv, "10.5f") + "\n")
    f.close()
    # final geometry
    f = open("FINAL/intcin", "w")
    f.write("TEXAS\n")
    for atom in final_cart_data:
        f.write("  " + atom[0]
                + format(float(atom[1]), "17.5f")
                + format(float(atom[2])*conv, "10.5f")
                + format(float(atom[3])*conv, "10.5f")
                + format(float(atom[4])*conv, "10.5f") + "\n")
    f.close()
    print("intcin generated")

    # run intc to generate intcfl
    # initial geometry
    rv = subprocess.run(["/home/cavanes1/col/Columbus/intc.x"],cwd="./0",capture_output=True)
    print("Initial intc output:")
    print(rv.stdout.decode('utf8'))
    # final geometry
    rv = subprocess.run(["/home/cavanes1/col/Columbus/intc.x"],cwd="./FINAL",capture_output=True)
    print("Final intc output:")
    print(rv.stdout.decode('utf8'))

# generate internal coordinates from Cartesian coordinates
# generate cart2intin
c2itxt = """ &input
    calctype='cart2int'
 /"""
f = open("0/cart2intin", "w")
f.write(c2itxt)
f.close()
f = open("FINAL/cart2intin", "w")
f.write(c2itxt)
f.close()
# generate dummy cartgrd
cartgrdtxt = """    0.739462D-07   0.107719D-01   0.359672D-02
  -0.637601D-07   0.573448D-03   0.113684D-01
  -0.408939D-07  -0.123975D-01  -0.519692D-02
   0.345788D-07  -0.168148D-02  -0.133691D-01
  -0.145659D-07   0.423517D-02   0.559146D-02
   0.106949D-07  -0.150151D-02  -0.199053D-02"""
f = open("0/cartgrd", "w")
f.write(cartgrdtxt)
f.close()
f = open("FINAL/cartgrd", "w")
f.write(cartgrdtxt)
f.close()
# run cart2int
rv = subprocess.run(["/home/cavanes1/col/Columbus/cart2int.x"],cwd="./0",capture_output=True)
rv2 = subprocess.run(["/home/cavanes1/col/Columbus/cart2int.x"],cwd="./FINAL",capture_output=True)
print("Initial cart2int output:")
print(rv.stdout.decode('utf8'))
print("Final cart2int output:")
print(rv2.stdout.decode('utf8'))

# read internal coordinate files
f = open("0/intgeom", "r")
init_int = f.readlines()
f.close()
init_int_data = []
for line in init_int:
    init_int_data.append(float(line))
init_int_data = np.array(init_int_data)
f = open("FINAL/intgeom", "r")
final_int = f.readlines()
f.close()
final_int_data = []
for line in final_int:
    final_int_data.append(float(line))
final_int_data = np.array(final_int_data)
print("Internal geometry files read")

# create interpolation vector
interpvec = final_int_data - init_int_data
print("Interpolation vector created")

# reverse cart2intin direction in 0
i2ctxt = """ &input
    calctype='int2cart'
 /"""
f = open("0/cart2intin", "w")
f.write(i2ctxt)
f.close()
print("cart2intin direction reversed in initial geometry directory")

# run COLUMBUS for each step
def slurmcop(source, target):
    str_source = str(source)
    str_target = str(target)
    # make list of bond length distances
    path = './'
    distances = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
    if '.git' in distances:
        distances.remove('.git')
    # if target already exists
    if str_target in distances:
        print("!!!!!!!!!!!!TARGET ALREADY EXISTS")
        os.system("rm -r " + str_target)
    # copy the source folder's contents
    os.system("cp -r " + str_source + " " + str_target)
    print("copied " + str_source + " to " + str_target)
    # remove old slurm output(s)
    path = './' + str_target + "/"
    flist = [file for file in os.listdir(path) if os.path.isfile(path+file)]
    for file in flist:
        if 'slurm' in file:
            os.system("rm " + path + file)
    # generate interpolated geometry as intgeomch
    f = open(str_target + "/intgeomch", "w")
    currgeom = init_int_data + interpvec*step*target
    for coord in currgeom:
        f.write(format(float(coord), "14.8f") + "\n")
    f.close()
    # run cart2int to convert to Cartesians
    rv = subprocess.run(["/home/cavanes1/col/Columbus/cart2int.x"],cwd="./"+str_target,capture_output=True)
    #print("cart2int output:")
    #print(rv.stdout.decode('utf8'))
    # replace geom with geom.new
    os.system("cp " + str_target + "/geom.new " + str_target + "/geom")
    # Move old directory's orbitals into new directory
    os.system("cp " + str_source + "/MOCOEFS/mocoef_mc.sp " + str_target)
    # Rename starting orbitals
    os.system("mv " + str_target + "/mocoef_mc.sp " + str_target + "/mocoef")
    # run Columbus interactivelyish
    rv = subprocess.run(["/home/cavanes1/col/Columbus/runc"],cwd="./"+str_target,capture_output=True)
    # save results to runls
    h = open(str_target + '/runls', "w")
    h.write(rv.stdout.decode('utf8'))
    h.close()
    # remove WORK directory
    os.system("rm -r " + str_target + "/WORK")

# run copyfunction automatically
for i in range(int(1/step)):
    slurmcop(i, i + 1)
