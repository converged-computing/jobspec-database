import os,sys

def generate_config_file( config_file, template_file, pars ):

    tf = open(template_file,'r')
    template = tf.read()

    out = open(config_file,'w')
    print>>out,template.format( **pars )
    out.close()

def get_params_fromfile( param_file, lineno ):

    parfile = open(param_file,'r')
    parlines = parfile.readlines()

    if lineno+1 >= len(parlines) or lineno<0:
        return None

    pars = parlines[lineno+1].split('\t')
    pardict = {"epsilon":float(pars[0]),
               "Y":float(pars[1]),
               "M_chi":float(pars[2]),
               "M_v":float(pars[3]),
               "alpha_D":float(pars[4])}

    return pardict

if __name__ == "__main__":

    pardict = get_params_fromfile( sys.argv[1], int(sys.argv[2]) )
    outdir = sys.argv[5]
    pardict["outdir"] = outdir
    if pardict is not None:
        generate_config_file( sys.argv[3], sys.argv[4], pardict )
    

