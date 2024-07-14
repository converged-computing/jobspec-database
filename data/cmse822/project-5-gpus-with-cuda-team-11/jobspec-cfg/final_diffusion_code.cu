#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <cmath>
#include <cassert>


#include <sys/time.h>

void get_walltime_(double* wcTime) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  *wcTime = (double)(tp.tv_sec + tp.tv_usec/1000000.0);
}

void get_walltime(double* wcTime) {
  get_walltime_(wcTime);
}

using namespace std;

const unsigned int NG = 2;

// Block sizes
//const unsigned int BLOCK_DIM_X = 128;
//const unsigned int BLOCK_DIM_X = 256;
//const unsigned int BLOCK_DIM_X = 512;
const unsigned int BLOCK_DIM_X = 1024;


__constant__ float c_a, c_b, c_c;

/******************************************************************************
  Error checking function for CUDA
 *****************************************************************************/
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
//https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/
//        finite-difference/finite-difference.cu
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

/******************************************************************************
  Do one diffusion step, on the host in host memory
 *****************************************************************************/
void host_diffusion(float* u, float *u_new, const unsigned int n,
                    const float dx, const float dt)
{
  //First, do the diffusion step on the interior points
    for(int i = NG; i < n-NG;i++){
        u_new[i] = u[i] + dt/(dx*dx) *(
                        - 1./12.f* u[i-2]
                        + 4./3.f * u[i-1]
                        - 5./2.f * u[i]
                        + 4./3.f * u[i+1]
                        - 1./12.f* u[i+2]);

   }

    //Apply the dirichlet boundary conditions
    u_new[0] = -u_new[NG+1];
    u_new[1] = -u_new[NG];

    u_new[n-NG]   = -u_new[n-NG-1];
    u_new[n-NG+1] = -u_new[n-NG-2];
}
/******************************************************************************
  Do one diffusion step, with CUDA
 *****************************************************************************/
__global__
void cuda_diffusion(float* u, float *u_new, const unsigned int n)
{
    //Do the diffusion
    //FIXME

    // get the global index in the 1-d grid of blocks which contain threads 
    int i = blockIdx.x*blockDim.x + threadIdx.x + 2;

    // constants already include the negative signs 
    u_new[i] = u[i] + ( c_a*u[i-2]
                       +c_b*u[i-1]
                       +c_c*u[i]
                       +c_b*u[i+1]
                       +c_a*u[i+2]);


    //Apply the dirichlet boundary conditions 

    if(i==2) { u_new[1] = -u_new[i] ; }
    if(i==3) { u_new[0] = -u_new[i];}

    if(i==n-3) {   u_new[n-2] = -u_new[i]; }
    if(i==n-4){    u_new[n-1] = -u_new[i];  }





}

/******************************************************************************
  Do one diffusion step, with CUDA, with shared memory
 *****************************************************************************/
__global__
void shared_diffusion(float* u, float *u_new, const unsigned int n)
{
    //Allocate the shared memory


    __shared__ float shared_m[BLOCK_DIM_X + 4];

    // local index for shared memory
    int s_index_ = threadIdx.x + 2;

    // global index of thread 
    int g_index_ = blockIdx.x*blockDim.x + s_index_;

    shared_m[s_index_] = u[g_index_];

    if(s_index_ < 4)
    {
         shared_m[s_index_ - 2] = u[g_index_ - 2];
    }
    else if(s_index_ >= blockDim.x - 4)
    {
       shared_m[s_index_ + 2] = u[g_index_ + 2];
    }
    __syncthreads();

    u_new[g_index_] = shared_m[s_index_] + ( c_a*shared_m[s_index_-2]
                          + c_b*shared_m[s_index_-1]
                          + c_c*shared_m[s_index_]
                          + c_b*shared_m[s_index_+1]
                          + c_a*shared_m[s_index_+2]);

    //Apply the dirichlet boundary conditions
    if(g_index_ < 4)
     {
        u_new[(g_index_ + 1)%2] = -u_new[g_index_];
     }
    if(g_index_ >= n - 4)
    {
     u_new[2*(n - 2)-(g_index_ + 1)] = -u_new[g_index_];
    }

}

/******************************************************************************
  Dump u to a file
 *****************************************************************************/
void outputToFile(string filename, float* u, unsigned int n)
{
    ofstream file;
    file.open(filename.c_str());
    file.precision(8);
    file << std::scientific;
    for(int i =0; i < n;i++){
        file<<u[i]<<endl;
    }
    file.close();
};

/******************************************************************************
  main
 *****************************************************************************/
int main(int argc, char** argv)
{
    //Number of steps to iterate
    //const unsigned int n_steps = 10;
    const unsigned int n_steps = 100;
    //const unsigned int n_steps = 1000000;

    //Whether and how ow often to dump data
    //const bool outputData = true;
    const bool outputData = false;
    const unsigned int outputPeriod = n_steps/10;

    //Size of u
    //const unsigned int n = (1<<11) +2*NG;
    //const unsigned int n = (1<<15) +2*NG;
    const unsigned int n = (1<<20) +2*NG;

    //Block and grid dimensions
    const unsigned int blockDim = BLOCK_DIM_X;
    const unsigned int gridDim = (n-2*NG)/blockDim;

    //Physical dimensions of the domain
    const float L = 2*M_PI;
    const float dx = L/(n-2*NG-1);
    const float dt = 0.25*dx*dx;

    //Create constants for 6th order centered 2nd derivative
    // NOTE: I added the appropriate negative signs here
    float const_a = -1./12.f * dt/(dx*dx);
    float const_b = 4./3.f  * dt/(dx*dx);
    float const_c = -5./2.f  * dt/(dx*dx);

    //Copy these the cuda constant memory
    //FIXME


    // Copy weights to device constants
    checkCuda(cudaMemcpyToSymbol(c_a, &const_a, sizeof(float),0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(c_b, &const_b, sizeof(float),0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(c_c, &const_c, sizeof(float),0, cudaMemcpyHostToDevice));

    //iterator, for later
    int i;

    //Create cuda timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Timing variables
    float milliseconds;
    double startTime,endTime;

    //Filename for writing
    char filename[256];

    //Allocate memory for the initial conditions
    float* initial_u = new float[n];

    //Initialize with a periodic sin wave that starts after the left hand
    //boundaries and ends just before the right hand boundaries
    for( i = NG; i < n-NG; i++)
    {
        initial_u[i] = sin( 2*M_PI/L*(i-NG)*dx);
    }
    //Apply the dirichlet boundary conditions
    initial_u[0] = -initial_u[NG+1];
    initial_u[1] = -initial_u[NG];

    initial_u[n-NG]   = -initial_u[n-NG-1];
    initial_u[n-NG+1] = -initial_u[n-NG-2];

    /**************************************************************************
    Test the host kernel for diffusion
    **************************************************************************/

    //Allocate memory in the host's heap
    float* host_u  = new float[n];
    float* host_u2 = new float[n];//buffer used for u_new

    //Initialize the host memory
    for( i = 0; i < n; i++)
    {
        host_u[i] = initial_u[i];
    }

    outputToFile("data/host_uInit.dat",host_u,n);

    get_walltime(&startTime);
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++){

        if(outputData && i%outputPeriod == 0)
        {   printf("ok \n");
            sprintf(filename, "/mnt/home/hejaseba/mpii/cuda_/project_5/host_u%08d.dat", i);
            outputToFile(filename,host_u,n);
        }

        host_diffusion(host_u, host_u2, n, dx, dt);

        //Switch the buffer with the original u
        float* tmp = host_u;
        host_u = host_u2;
        host_u2 = tmp;

    }
    get_walltime(&endTime);

    cout << "Host function took: " << (endTime-startTime)*1000./n_steps
         << " ms per step" << endl;

    outputToFile("data/host_uFinal.dat", host_u, n);



  /**************************************************************************
    Test the cuda kernel for diffusion
    **************************************************************************/
    //Allocate a copy for the GPU memory in the host's heap
    float* cuda_u  = new float[n];

    //Initialize the cuda memory
    for( i = 0; i < n; i++)
    {
        cuda_u[i] = initial_u[i];
    }
    outputToFile("data/cuda_uInit.dat", cuda_u, n);

    //Allocate memory on the GPU
    float* d_u, *d_u2;
    //FIXME Allocate d_u,d_u2 on the GPU, and copy cuda_u into d_u

    // allocating data on the device's (gpu's) memory
    checkCuda(cudaMalloc((void**)&d_u, n*sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_u2,n*sizeof(float)));

    // host to device copy
    checkCuda(cudaMemcpy(d_u, cuda_u, n*sizeof(float),cudaMemcpyHostToDevice));

    cudaEventRecord(start);//Start timing
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++)
    {
        if(outputData && i%outputPeriod == 0)
        {
            //Copy data off the device for writing
            sprintf(filename, "/mnt/home/hejaseba/mpii/cuda_/project_5/cuda_u%08d.dat", i);
            //FIXME

            // device to host copy
            checkCuda(cudaMemcpy(cuda_u, d_u, n*sizeof(float), cudaMemcpyDeviceToHost));

            outputToFile(filename,cuda_u,n);
        }

        //Call the cuda_diffusion kernel
        //FIXME

        cuda_diffusion<<<gridDim, blockDim>>>(d_u, d_u2, n);

        //Switch the buffer with the original u
        //FIXME

        float* tmp = d_u;

        d_u = d_u2;
        d_u2 = tmp;
    }
    cudaEventRecord(stop);//End timing

    //Copy the memory back for one last data dump
    sprintf(filename, "data/cuda_u%08d.dat", i);
    //FIXME


    // device to host copy solution
    checkCuda(cudaMemcpy(cuda_u, d_u, n*sizeof(float),cudaMemcpyDeviceToHost));
    outputToFile("data/cuda_uFinal.dat",cuda_u,n);

    //Get the total time used on the GPU
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Cuda Kernel took: " << milliseconds/n_steps
         << " ms per step" << endl;





   /**************************************************************************
    Test the cuda kernel for diffusion with shared memory
    **************************************************************************/

    //Allocate a copy for the GPU memory in the host's heap
    float* shared_u  = new float[n];

    //Initialize the cuda memory
    for( i = 0; i < n; i++)
    {
        shared_u[i] = initial_u[i];
    }
    outputToFile("data/shared_uInit.dat",shared_u,n);

    //Copy the initial memory onto the GPU
    //FIXME copy shared_u to d_u

    checkCuda(cudaMemcpy(d_u, shared_u, n*sizeof(float),cudaMemcpyHostToDevice));


    cudaEventRecord(start);//Start timing
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++)
    {
        if(outputData && i%outputPeriod == 0)
        {
            //Copy data off the device for writing
            sprintf(filename,"/mnt/home/hejaseba/mpii/cuda_/project_5/shared_u%08d.dat",i);
            //FIXME




            checkCuda(cudaMemcpy(shared_u, d_u, n*sizeof(float), cudaMemcpyDeviceToHost));


            outputToFile(filename,shared_u,n);
        }


        shared_diffusion<<<gridDim, blockDim>>>(d_u, d_u2, n);



        float* tmp = d_u;
        d_u = d_u2;
        d_u2 = tmp;

    }
    cudaEventRecord(stop);//End timing


    sprintf(filename,"data/shared_u%08d.dat",i);
    //FIXME

    // Copy final solution back to the device
    checkCuda(cudaMemcpy(shared_u, d_u, n*sizeof(float),cudaMemcpyDeviceToHost));

    //Get the total time used on the GPU
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Shared Memory Kernel took: " << milliseconds/n_steps
         << " ms per step" << endl;



    /**************************************************************************
    Test the cuda kernel for diffusion, with excessive memcpys
    **************************************************************************/

    //Initialize the cuda memory
    for( i = 0; i < n; i++)
    {
        shared_u[i] = initial_u[i];
    }

    cudaEventRecord(start);//Start timing
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++)
    {

        checkCuda(cudaMemcpy(d_u, shared_u, n*sizeof(float),cudaMemcpyHostToDevice));

        shared_diffusion<<<gridDim, blockDim>>>(d_u, d_u2, n);

        checkCuda(cudaMemcpy(shared_u, d_u2, n*sizeof(float),cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(stop);//End timing

    //Get the total time used on the GPU
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Excessive cudaMemcpy took: " << milliseconds/n_steps
         << " ms per step" << endl;

    // freeing memory on host
    delete[] initial_u;
    delete[] host_u;
    delete[] host_u2;

    delete[] cuda_u;
    delete[] shared_u;


    // freeing memory on device
    checkCuda( cudaFree(d_u) );
    checkCuda( cudaFree(d_u2) );
}
