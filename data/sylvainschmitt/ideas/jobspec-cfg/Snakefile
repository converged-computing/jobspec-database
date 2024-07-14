configfile: "config/config.yml"

# rules #
rule all:
   input:
      expand("results/data/{source}.nc",
             source=["modis", "chirps"])

## rules ##
include: "rules/get_gadm.py"
include: "rules/get_area.py"
include: "rules/get_tmf.py"
include: "rules/get_chirps.py"
include: "rules/get_modis.py"
