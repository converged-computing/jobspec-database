from pyCosito.main import *
import os
import shutil
import logging

# Configurar el nivel de registro de la raíz del logger
logging.getLogger().setLevel(logging.ERROR)

# Configurar el nivel de registro de todos los loggers hijos de fireworks
logging.getLogger('fireworks').setLevel(logging.ERROR)

# Configurar el formato de los mensajes de registro
logging.basicConfig(format='%(levelname)s')

host = '10.150.0.14'
port = 25432
software = "PWDFT"
nnodes = 8

## Segun el articulo 1000 ev es suficiente
## 1000 eV = 36.7493


list_moles = ['LiH','BeH','CH','CH2_s1A1d','CH2_s3B1d','CH3','CH4','NH','NH2',
            'NH3','OH','H2O','HF','SiH2_s1A1d','SiH2_s3B1d','SiH3','SiH4','PH2',
            'PH3','SH2','HCl','Li2','LiF','C2H2','C2H4','C2H6','CN','HCN','CO',
            'HCO','H2CO','CH3OH','N2','N2H4','NO','O2','H2O2','F2','CO2','Na2',
            'Si2','P2','S2','Cl2','NaCl','SiO','CS','SO','ClO','ClF','Si2H6',
            'CH3Cl','CH3SH','HOCl','SO2']

if software == "XTB":
    conf = {"method" : "GFN1-xTB"}
elif software == "PWDFT":
    conf = {'nwpw':{'cutoff':37.0, 'xc':'PBE'}}
elif software == "Espresso":
    conf = {}
else:
    conf = {}

myWF = AtomizationWorkFlow(os.getcwd(), software=software, nworkers=nnodes,
                 software_kwargs=conf, moleculeList=list_moles,
                 name="MyWorkFlow", host=host, port=port, nameDB='myXTBdb')

myWF.runWF()


print("  Remove launcher subdir")
files = os.listdir()

for file in files:
    if file.startswith("launcher"):
        shutil.rmtree(file)

