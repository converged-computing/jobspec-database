# Solvation Energy Values are features
import numpy as np
import os 
pqr_filepaths = []
for protein in os.listdir('/users/esliheet/esliheet/predictSE/v2007'):
   for file in os.listdir('/users/esliheet/esliheet/predictSE/v2007/' + protein):
       if file.endswith('.pqr'):
           pqr_file = '/users/esliheet/esliheet/predictSE/v2007/' + protein + '/' + protein + '_AMBER'
           pqr_filepaths.append(pqr_file)

# Generate list of commands
string1 = 'mpirun -np 64 ./tabipb.exe \''
string2 = '\' 16 1 80 0.15 3 100 0.8'
tabi_commands = [(string1+s+string2) for s in pqr_filepaths]

counter = 0
labels = []
dir = "/users/esliheet/esliheet/tabipb_cyclic_precond"
os.chdir(dir)
#os.system('make clean')
os.system('make')
for command in tabi_commands:
    counter = counter + 1
    print("counter:", counter)
    #os.system('make clean')
    #os.system('make')
    os.system(command)
    label = np.loadtxt('output.txt')
    print("label:", label)
    labels.append(label)
    
#Convert features to numpy array
y = np.vstack(labels)
print("y:", y)
dir = "/users/esliheet/esliheet/predictSE"
os.chdir(dir)
print("y shape: ", y.shape)
np.save("y", y)
