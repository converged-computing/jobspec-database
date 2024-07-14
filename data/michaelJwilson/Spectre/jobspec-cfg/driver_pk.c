#define   KBIN_NO     40
#define   FOLDFACTOR 2.0       
 
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>
#include "omp.h"

#include "header.h"
#include "header_pk.h"
#include "cosmology_planck15.h"
#include "struct_regress.h"
#include "walltime.c"
#include "Initialise.c"
#include "CoordinateCalc.c"
#include "MultipoleCalc.c"
#include "ngp.c"
#include "CloudInCell.c" 
#include "overdensity_calc.c"
#include "FFTw.c"
#include "assign_pkmemory.c"
#include "assign_binnedpk_memory.c"


int main(int argc, char **argv){  
  (void) argc;                                              // escape compiler unused variable warning. 

  thread                    =                   1; 
   
  max_gals                  =             3238855;
  
  //  outputdir                 = getenv("outputdir");  

  sprintf(filepath,      "/global/homes/m/mjwilson/UNIT/sim/HOD_Shadab/HOD_boxes/redshift0.9873/UNIT_DESI_Shadab_HOD_snap97_ELG_v0.txt");

  fkpPk                     =    8000.0;                    // [h^-1 Mpc]^3.  Stefano: 4000 [h^-1 Mpc]^3.

  fft_size                  =      512;                     // Worker 46 works up to 1024. 
  
  logk_min                  =      -2.0;
  logk_max                  =   0.00000;                    // k = 1 hMpc^{-1} :  0.00000;  k = 3 hMpc^{-1} :  0.47712;  k = 4 hMpc^{-1} : 0.60206 
  
  start_walltime();
  
  //  printf_branch();
  
  fftw_init_threads();
  
  fftw_plan_with_nthreads(omp_get_max_threads());        // Maximum number of threads to be used; use all openmp threads available.  
  
  
  Initialise();                                          // Initialise grid, FFT params and random generation.
    
  prep_CatalogueInput_500s(max_gals);                    // Max. number of gals of ALL mocks (& data) analysed simultaneously is `hard coded' (with some contingency).  
  
  prep_x2c();                                            // Memory for overdensity, smooth_overdensity and H_k; either double or fftw_complex.

  prep_pkRegression();                                   // set k binning interval arrays.  
    
  prep_r2c_modes(&flat,                    1.0);         // unfolded.
  //  prep_r2c_modes(&half,             FOLDFACTOR);         // one fold.
  //  prep_r2c_modes(&quart, FOLDFACTOR*FOLDFACTOR);         // two folds.

  
  //  regress set[3] = {flat, half, quart};
  regress set[1] = {flat};      

  
  CatalogueInput_500s(max_gals);  // mocks 1 to 153 are independent. 
    
  calc_overdensity(max_gals);

  // for(fold=0; fold<3; fold++){
  PkCalc(&set[0], 0);
  //  }
  
  // MPI_Finalize();
  
  printf("\n\n");
  
  exit(EXIT_SUCCESS);
}
