// Roberto Alfieri, Marco Borelli, Roberto De Pietri, Paolo Leoni
// University of Parma - INFN
// Life_hpc

char version[]="2014.02.23";
int DEBUG=1;

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>    //usleep
#include <sys/time.h>  //gettimeofday
#include <string.h>    //strcpy

// #include "grid.hh" //grid class

#ifdef MPI
#include <mpi.h>
#endif /* MPI */


#ifdef OMP
#include <omp.h>
#endif /* OMP */

#if _OPENACC
#include <openacc.h>
#endif

///////////////////////////


void options(int argc, char * argv[]) ;
void usage(char * argv[]);
void copy_border(int rmin, int rmax, int cmin, int cmax, double ** grid);
void init_grid(int nrows, int rmin, int rmax, int ncols, double *** grid, double prob);
void allocate_grid(int,  int , double *** grid);
void randomize_grid(int, int , double ** grid, double base_life);
double rand_double();
void  do_step(int rmin, int rmax, int cmin, int cmax, double ** grid, double ** next_grid);
void  do_display(int rmin, int rmax, int cmin, int cmax, double ** grid);
void grid_copy(int rmin, int rmax, int cmin, int cmax, double ** grid, double ** next_grid);
void random_initByTime(int rank) ;
void clearscreen();
void copy_border(int rmin, int rmax, int cmin, int cmax, double ** grid);

void init_GPU();

 int mygpu=0; //default GPU

 int nsteps=1000;       //!< Number of Steps 
 int ncols=80;          //!< Number of Columns 
 int nrows=40;          //!< Number of rRows 
 double base_life=0.2;  //!< Base probability
 char datafile[80]=""; 
 char hostname[80];
 long datasize;
 int ncomp=1000;            //!< Computation load
 double sum=0.0;

 

 double ** grid;              
 double ** next_grid;

 double *A, *B; // array for computation

// OMP
 int num_threads=1;  //default omp threads
 //MPI
 char mpirank_name[4]; 
 char mpisize_name[4];

/////////// MPI /////////

int mpi_rank=0, mpi_size=0;  
int prev_rank=0, next_rank=0;  


#ifdef MPI
MPI_Request request[4]; 
MPI_Status  status[4];
#endif /* MPI */


/////////// OMP //////////

int omp_rank=0, omp_size=1; 

/////////////////////////////////////////

int main(int argc, char ** argv) {


 double  ta, tb, tc;
 struct timeval tempo ;

 options(argc, argv);         /* optarg management */ 

#ifdef OMP

	omp_set_num_threads(num_threads);  // if is set (>0) keep this value else use t=1

#endif /* OMP */


gethostname(hostname, 100);  /* get hostname */
datasize=nrows*ncols*sizeof(double) ;     /* tot number of bytes */ 
	

double *ptr;

grid      = (double **)  malloc ( sizeof(ptr) * (nrows+2)  );  // init grid
next_grid = (double **)  malloc ( sizeof(ptr) * (nrows+2)  );  // init next_grid

int i,j;

#ifdef COMP

//A = (double *) malloc ( sizeof(ptr) * ncomp ); 
//B = (double *) malloc ( sizeof(ptr) * ncomp );

posix_memalign((void*)&(A), 64, ncomp*sizeof(double)); //memory alignment
posix_memalign((void*)&(B), 64, ncomp*sizeof(double)); //memory alignment

for (i=0; i< ncomp; i++) A[i]=rand_double();
for (i=0; i< ncomp; i++) B[i]=rand_double();



#endif //////// A[] and B[] allocation


#ifdef MPI

	MPI_Init(&argc, &argv); 
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	prev_rank = (mpi_rank-1+mpi_size) % mpi_size;
	next_rank = (mpi_rank+1) % mpi_size;

	mygpu = mpi_rank%2;

	sprintf(mpirank_name,"%d",mpi_rank);  /* convert integer to string */
	sprintf(mpisize_name,"%d",mpi_size);  /* convert integer to string */ 

#endif /* MPI */

	if (!strcmp(datafile,"")) sprintf(datafile,"life_%ld_%d_%d.dat",datasize,mpi_rank,mpi_size);  /* file name */	

//initialize

	if (DEBUG==1) fprintf(stdout,"\n%s-%d MPI_INIT mpi_size:%d omp_size:%d ncols:%d nrows:%d nsteps:%d file:%s debug:%d\n", hostname, mpi_rank, mpi_size, num_threads, ncols,nrows,nsteps,datafile, DEBUG);

#ifdef COMP
	if (DEBUG==1) fprintf(stdout,"\nComp load: %d\n",ncomp);
#else
	if (DEBUG==1) fprintf(stdout,"\nComp load: DISABLED\n");
#endif
 
	if (DEBUG==1) fprintf(stderr, "\n%s-%d ALLOCATE MEMORY  (%ld grid + %ld new_grid = %ld bytes ) \n", hostname,mpi_rank, datasize, datasize, datasize*2);
			   													   

	if (DEBUG==1) fprintf(stderr, "%s-%d  %d iterations - Start timer\n",hostname ,mpi_rank, nsteps);
	if (DEBUG==1) fprintf(stderr,"%s-%d  START_MEM_ALLOC \n", hostname,mpi_rank); 		

	gettimeofday(&tempo,0);  ta=tempo.tv_sec+(tempo.tv_usec/1000000.0); // Save current time in TA

	init_grid( nrows, 1, nrows+1, ncols, &grid, base_life);
	init_grid( nrows, 1, nrows+1, ncols, &next_grid, 0);

	gettimeofday(&tempo,0);  tb=tempo.tv_sec+(tempo.tv_usec/1000000.0); // Save current time in TB

	if (DEBUG==1) fprintf (stderr, "%s-%d  END_MEM_ALLOC %f sec - START_DO_STEPS %d rows:%d-%d cols:%d-%d\n", hostname,mpi_rank, tb-ta, nsteps, 0,nrows+1,0,ncols+1);
    

 // #pragma omp parallel  private(omp_rank)

  {
	#ifdef OMP 

	omp_rank=omp_get_thread_num();
	omp_size=omp_get_num_threads();

	#endif /* OMP */	

	//int rmin=(nrows*omp_rank)/omp_size+1;    //first row
	//int rmax=(nrows*(omp_rank+1))/omp_size;  //last row 
	//int cmin=1;      // first col
	//int cmax=ncols;  // last col

	int rmin=1;
	int cmin=1;

	int rmax=nrows;
	int cmax=ncols;

	int k;	
	
#if _OPENACC
	init_GPU();
#endif

#ifdef MIC
	init_MIC();
#endif

#if _OPENACC	
	#pragma acc data async(3) copyin(A[ncomp],B[ncomp],grid[nrows+2][ncols+2]) create(next_grid[nrows+2][ncols+2],sum)
#endif
	for(k=1; k<nsteps; k++)
		{

		do_step(rmin,rmax,cmin,cmax, grid, next_grid);
		
		grid_copy(rmin,rmax,cmin,cmax, grid, next_grid);

  		#pragma omp single
			{
			copy_border(1, nrows, 1, ncols,  grid);

			if (DEBUG==2) do_display (1, nrows, 1, ncols,  grid);

			}//end omp single
		}

	if (DEBUG==1) fprintf(stderr,"%s-%d %d/%d OMP-PARALLEL STOP\n", hostname,mpi_rank,omp_rank,omp_size); 	

  } // end openMP parallel


    gettimeofday(&tempo,0); tc=tempo.tv_sec+(tempo.tv_usec/1000000.0); // Save current time in TC


    if (DEBUG>0) fprintf(stderr,"%s-%d - Finalize  - %f sec  \n" , hostname,mpi_rank, tb-ta);

if (mpi_rank==0)
    if (DEBUG<2) fprintf(stderr,"%d %d %d %d %d %d %f %f %f # %s \n" ,  mpi_size, omp_size, ncols, nrows, nsteps, ncomp,  tb-ta, tc-tb, tc-ta, hostname );
else
    if (DEBUG<2) fprintf(stderr,"#%d %d %d %d %d %d %f %f %f # %s \n" ,  mpi_size, omp_size, ncols, nrows, nsteps, ncomp,  tb-ta, tc-tb, tc-ta, hostname );


#ifdef MPI

    MPI_Finalize();

#endif /* MPI */ 

    return 0;
}



////////////////////////////// border //////////////////////////////


void  copy_border(int rmin, int rmax, int cmin, int cmax, double ** grid) {

  int i;

// copy rows
#if _OPENACC
	#pragma acc kernels present(grid[nrows+2][ncols+2])
	#pragma acc loop independent
#endif
  for (i = cmin - 1; i <= cmax + 1; ++i) {

	    grid[rmin-1][i] = grid[rmax][i];
	    grid[rmax+1][i] = grid[rmin][i];
	}

#ifndef MPI


// copy cols
#if _OPENACC
	#pragma acc kernels present(grid[nrows+2][ncols+2])
	#pragma acc loop independent
#endif
  for (i = rmin - 1; i <= rmax + 1; ++i) {
    grid[i][cmin-1] = grid[i][cmax];
    grid[i][cmax+1] = grid[i][cmin];

	}

#else


////////////////////////////// MPI 

int tag = 999;



// create 2 temp columns 

double *col_send, *col_recv; 

col_send= (double *)  malloc (  sizeof (double) * (nrows+2) ) ;
col_recv= (double *)  malloc (  sizeof (double) * (nrows+2) ) ;

for (i=0; i<nrows+2; i++) col_send[i]=grid[i][ncols];  //Copy Col n to send buff

 MPI_Sendrecv(col_send, nrows + 2, MPI_DOUBLE, next_rank, tag, //send
              col_recv, nrows + 2, MPI_DOUBLE, prev_rank, tag, //recv
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//printf("MPIsendrecv send to %d - recv from %d\n", next_rank, prev_rank); 

for (i=0; i<nrows+2; i++)  grid[i][0]=col_recv[i] ;  //Copy recv buff to Col 0
for (i=0; i<nrows+2; i++) col_send[i]=grid[i][1];  // Copy Col 1 to send buff

 MPI_Sendrecv(col_send, nrows + 2, MPI_DOUBLE, prev_rank, tag, // send
              col_recv, nrows + 2, MPI_DOUBLE, next_rank, tag, // recv 
              MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//printf("MPIsendrecv send to %d - recv from %d\n", prev_rank, next_rank);

for (i=0; i<nrows+2; i++) grid[i][ncols+1]=col_recv[i];  //copy recv buff to Col n+1
 
free(col_send);
free(col_recv);

MPI_Barrier(MPI_COMM_WORLD);

#endif /* MPI */

/////////////////////////////////////////////////

 return ;

}


////////////////////////// options //////////////////////////////////////

void options(int argc, char * argv[]) {

  int i;
   while ( (i = getopt(argc, argv, "W:vc:r:s:d:ht:f:C:n:G:")) != -1) {
        switch (i) {
        case 'c':  ncols       = strtol(optarg, NULL, 10);  break;
	case 'G':  mygpu       = strtol(optarg, NULL, 10);  break;
        case 'r':  nrows       = strtol(optarg, NULL, 10);  break;
        case 's':  nsteps      = strtol(optarg, NULL, 10);  break;
        case 'n':  ncomp       = strtol(optarg, NULL, 10);  break;
        case 't':  num_threads = strtol(optarg, NULL, 10);  break;
        case 'd':  DEBUG       = strtol(optarg, NULL, 10);  break;			
        case 'h':  usage(argv); exit(1);
        case 'v':  printf("%s version %s\n",argv[0],version); exit(1);
        case '?':  usage(argv); exit(1);
        default:   usage(argv); exit (1);
        }
    }
}

////////////////////////// usage //////////////////////////////////////

void usage(char * argv[])  {

  printf ("\n%s [-c ncols] [-r nrows] [-t num_thr] [-s nsteps] [-G ngpu] [-d debug] [-v] [-h]",argv[0]); 
  printf ("\n -d <0|1|2> : <no output | debug info (default) | display interactively> ");   
  printf ("\n -s <int>   : steps  (step num. default=1000)"); 
  printf ("\n -n <int>   : Computation load  (default=1000)"); 
  printf ("\n -G <int>   : GPU (default 0)");
  printf ("\n -t <int>   : Threads num ( default = OMP_NUM_THREADS if set, otherwise = cores num"); 
  printf ("\n -v   : version "); 
  printf ("\n -h   : help "); 
  printf ("\n"); 

}


////////////////////////// grid_copy //////////////////////////////////////

void grid_copy(int rmin, int rmax, int cmin, int cmax, double ** grid, double ** next_grid) 
{
 int i,j;
 #pragma omp parallel for private(i,j)
#if _OPENACC
	#pragma acc kernels async(2) present(grid[nrows+2][ncols+2],next_grid[nrows+2][ncols+2])
	#pragma acc loop gang independent
#endif
 for (i=rmin;i<=rmax;i++)
 {
#if _OPENACC
     #pragma acc loop vector independent
#endif
     for (j=cmin;j<=cmax;j++)
	{
         grid[i][j]=next_grid[i][j];  
	}
 }

}

////////////////////////////// do_step //////////////////////////////

void do_step(int rmin, int rmax, int cmin, int cmax, double ** grid, double ** next_grid)

 {
  int k,l,j,i;
  double neighbors=0.0;
  #pragma omp parallel for private(i,j,k) reduction(+: sum)
#if _OPENACC
	#pragma acc kernels async(1) present(grid[nrows+2][ncols+2],next_grid[nrows+2][ncols+2],A[ncomp],B[ncomp],sum)
	#pragma acc loop gang independent
#endif
  for (i=rmin; i<=rmax; i++) {
#if _OPENACC
	#pragma acc loop vector independent
#endif

       for (j=cmin; j<=cmax; j++) {
	#ifdef COMP
	        #pragma ivdep
		#pragma vector aligned
		#if _OPENACC
			#pragma acc loop reduction(+: sum)
		#endif
				
 		for (k=0; k < ncomp; k++) {
			sum += A[k] + B[k];
					}
	#endif
		
	       neighbors=grid[i+1][j+1] + grid[i+1][j] + grid[i+1][j-1] + grid[i][j+1] + grid[i][j-1] + grid[i-1][j+1]+grid[i-1][j]+grid[i-1][j-1];
	       if ( ( neighbors > 3.0 ) || ( neighbors < 2.0 ) )
                  next_grid[i][j] = 0.0;
	       else if ( neighbors == 3.0 )
                  next_grid[i][j] = 1.0;
               else
                  next_grid[i][j] =  grid[i][j];
		}
	}

}

/////////////////////////// do_display ////////////////////////////////////

void do_display(int rmin, int rmax, int cmin, int cmax, double ** grid)

{
  int i,j; 
  int delay=200000;       /* usec sleep in do_display */

  clearscreen();
  for(i=cmin;i<=cmax;i++) printf("-"); printf ("\n");
    for (i=rmin;i<=rmax;i++) { 
		for (j=cmin;j<=cmax;j++)
			if (grid[i][j]==0) printf(" "); 
				else printf ("x"); 
		printf ("\n");
	}

   usleep(delay);
}


/////////////////////////// allocate_grid ////////////////////////////////////

void allocate_grid(int nrows, int ncols, double *** grid){

    int *a,i,j;
    (*grid) =  (double **)    malloc ( sizeof(double *) * (nrows+2) );

    for (i=0; i<nrows+2;i++) {
        (*grid)[i] =  (double *)  malloc (  sizeof (double) * (ncols+2) ) ;
        for (j=0;j<ncols+2;j++) {
            (*grid)[i][j]=0;    
        }
    }
}

/////////////////////////// init_grid ////////////////////////////////////

void init_grid(int nrows, int rmin, int rmax, int ncols, double *** grid, double prob){


    int i,j; 

    random_initByTime(rand()%100);

    if (rmin==1) rmin=0;
    if (rmax==nrows) rmax=nrows+1;  

    printf("init-grid %d-%d %d %f\n",  rmin,rmax,ncols,prob);

    for (i=rmin; i<rmax+1;i++) 
  	{

	(*grid)[i] =  (double *)  malloc ( sizeof (double) * (ncols+2) );
	if (prob)  
		for ( j=0;j<=ncols+1;j++)  
			if ( rand_double() <prob ) (*grid)[i][j]=1; 
				else 	 (*grid)[i][j]=0;	
//        for (j=0;j<ncols+2;j++) (*grid)[i][j]=0;       

	}
}


/////////////////////////// randomize_grid ////////////////////////////////////

void randomize_grid(int nrows, int ncols, double ** grid, double prob){

  int i,j; 

  random_initByTime(rand()%100) ;
  for ( i=1;i<=nrows;i++)
    for ( j=1;j<=ncols;j++)
      if (rand_double() < prob) 
	grid[i][j] = 1.0;
      else
	grid[i][j] = 0.0;

}

/////////////////////////// rand_double ////////////////////////////////////

double rand_double() {

    return (double)rand()/(double)RAND_MAX;

}

/////////////////////////// random_iniyByTime ////////////////////////////////////

void random_initByTime(int rank) {

    time_t ltime;
    time(&ltime);

    //srand((unsigned) ltime + 100*rank);

    srand((unsigned) 123);

}

/////////////////////////// clearscreen ////////////////////////////////////

void clearscreen() {

if (system( "clear" )) system( "cls" );

}


//////////////////////// init_GPU ///////////////////////////////////////////

#if _OPENACC
void init_GPU() {

	acc_init(acc_device_nvidia);
	int myrealgpu, num_devices;
	acc_device_t my_device_type;

	//my_device_type = acc_device_cuda; //uncomment this if you are using CAPS
	my_device_type = acc_device_nvidia; //comment this if you are using CAPS

	acc_set_device_type(my_device_type) ;
	
	num_devices = acc_get_num_devices(my_device_type) ;
	if (DEBUG==1) fprintf(stderr,"\nNumber of devices available: %d \n",num_devices);
	
	acc_set_device_num(mygpu,my_device_type);
	if (DEBUG==1) fprintf(stderr,"Trying to use GPU: %d\n",mygpu);
	
	myrealgpu = acc_get_device_num(my_device_type);
	if (DEBUG==1) fprintf(stderr,"Actually I am using GPU: %d\n\n",myrealgpu);

	if(mygpu != myrealgpu) {
		if (DEBUG==1) fprintf(stderr,"I cannot use the requested GPU: %d\n",mygpu);
	exit(1);
	}
}
#endif
