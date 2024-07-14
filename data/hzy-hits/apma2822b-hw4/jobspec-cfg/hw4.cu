#include <iostream>
#include <chrono>
#include <cmath>
#include "debug.cuh"
#include <memory>
#include <fstream>
#include <random>
const int warpSize = 32;
const int BLOCK_SIZE = 256; // 设置固定的block大小为1024

__device__ double atomicAddDouble(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void dotProductKernel(double *x, double *y, int n, double *partialSums)
{
    unsigned FULL_MASK = 0xffffffff;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    __shared__ double warpPartialSum[warpSize]; // 每个warp一个shared memory位置

    double sum = 0;
    if (index < n)
    {
        sum = x[index] * y[index];
    }
    else
    {
        sum = 0;
    }

    // Warp内部求和
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    // 每个warp的第一个线程将结果写入共享内存
    if (lane == 0)
    {
        warpPartialSum[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();

    // block内的第一个warp将所有warp的结果汇总
    if (threadIdx.x < warpSize)
    {
        sum = (threadIdx.x < blockDim.x / warpSize) ? warpPartialSum[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
    }

    // 将block的结果写入全局内存
    if (threadIdx.x == 0)
    {
        partialSums[blockIdx.x] = sum;
    }
}

__global__ void MatrixVectorProduct(double *matrix, double *vector, double *result, int rows, int cols)
{
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadIdx / warpSize;    // 计算全局warp ID
    int warpCount = (blockDim.x) / warpSize;    // 计算单个warp数量
    int rowsPerWarp = max(1, rows / warpCount); // 每个warp处理的行数
    extern __shared__ double shared_vector[];
    int threadId = threadIdx.x;
    if (threadId < cols)
    {
        shared_vector[threadId] = vector[threadId];
    }
    // __shared__ double shared_vector[BLOCK_SIZE];
    // for (int i = 0; i < cols; i += warpSize)
    // {
    //     if (threadIdx.x + i < cols)
    //     {
    //         shared_vector[threadIdx.x + i] = vector[threadIdx.x + i];
    //     }
    // }
    __syncthreads();
    int extraRows = rows % warpCount; // 不能均匀分配的额外行数

    int startRow = warpId * rowsPerWarp + min(warpId, extraRows);
    int endRow = startRow + rowsPerWarp + (warpId < extraRows ? 1 : 0);

    for (int row = startRow; row < endRow; row++)
    {
        double sum = 0;
        for (int col = threadIdx.x % warpSize; col < cols; col += warpSize)
        {
            sum += matrix[row * cols + col] * shared_vector[col];
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x % warpSize == 0)
        {
            atomicAddDouble(&result[row], sum);
        }
    }
}

void init_matrix(std::vector<double> &matrix, int row, int col)
{
    matrix.clear();
    std::random_device rd;  // 随机数种子
    std::mt19937 gen(rd()); // 使用Mersenne Twister算法的生成器
                            // 定义[0, 1)范围的均匀分布
    std::uniform_real_distribution<> distrib(0.0, 0.1);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {

            matrix.push_back(distrib(gen));

            // if (i == j)
            // {
            //     matrix.push_back(1);
            // }
            // else
            // {
            //     matrix.push_back(0);
            // }
        }
    }
}
void init_vector(std::vector<double> &vector, int row)
{
    vector.clear();
    std::random_device rd;  // 随机数种子
    std::mt19937 gen(rd()); // 使用Mersenne Twister算法的生成器

    // 定义[0, 1)范围的均匀分布
    std::uniform_real_distribution<> distrib(0.0, 0.01);

    for (int i = 0; i < row; i++)
    {
        vector.push_back(distrib(gen));
        // vector.push_back(1);
    }
}
void init_result(std::vector<double> &result, int row)
{
    result.clear();
    for (int i = 0; i < row; i++)
    {
        result.push_back(0);
    }
}
void initCudaMemory(double **d_ptr, const std::vector<double> &host_data)
{
    size_t size = host_data.size() * sizeof(double);
    CHECK(cudaMalloc(d_ptr, size));                                            // 在GPU上分配内存
    CHECK(cudaMemcpy(*d_ptr, host_data.data(), size, cudaMemcpyHostToDevice)); // 复制数据到GPU
}

void processMatrixInStreams(const std::vector<double> &matrix,
                            const std::vector<double> &vector,
                            std::vector<double> &result,
                            int M, int numRows, int numCols, float &cputime, float &gputime)
{
    // 确保矩阵的大小适合分块
    int rowsPerBlock = numRows / M;  // 每个块的行数
    int remainingRows = numRows % M; // 剩余的行数

    // 创建并初始化CUDA流
    cudaStream_t streams[M];
    for (int i = 0; i < M; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // 分配设备内存
    double *d_matrix, *d_result, *d_vector;
    cudaMalloc(&d_matrix, matrix.size() * sizeof(double));
    cudaMalloc(&d_vector, vector.size() * sizeof(double));
    cudaMalloc(&d_result, result.size() * sizeof(double));

    cudaHostRegister(const_cast<double *>(matrix.data()), matrix.size() * sizeof(double), 0x00);
    cudaHostRegister(const_cast<double *>(vector.data()), vector.size() * sizeof(double), 0x00);
    cudaHostRegister(result.data(), result.size() * sizeof(double), cudaHostRegisterDefault);

    // 复制数据到设备
    cudaMemcpy(d_vector, vector.data(), vector.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    // 在每个流上启动核函数
    for (int i = 0; i < M; ++i)
    {
        int startRow = i * rowsPerBlock;
        int rowsToProcess = (i == M - 1) ? rowsPerBlock + remainingRows : rowsPerBlock; // 最后一个块处理额外的行数
        cudaMemcpyAsync(d_matrix + startRow * numCols,
                        matrix.data() + startRow * numCols,
                        rowsToProcess * numCols * sizeof(double),
                        cudaMemcpyHostToDevice, streams[i]);
        MatrixVectorProduct<<<1, BLOCK_SIZE, numCols * sizeof(double), streams[i]>>>(d_matrix + startRow * numCols,
                                                                                     d_vector, d_result + startRow, rowsToProcess, numCols);
        cudaMemcpyAsync(result.data() + startRow,
                        d_result + startRow,
                        rowsToProcess * sizeof(double),
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // 等待所有流完成
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaEventRecord(stop);
    std::chrono::duration<double> duration = end - cpu_start;
    cputime = 0;
    cputime = duration.count() * 1000;
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    gputime = milliseconds;
    // 释放资源
    for (int i = 0; i < M; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
    cudaHostUnregister(const_cast<double *>(matrix.data()));
    cudaHostUnregister(const_cast<double *>(vector.data()));
    cudaHostUnregister(result.data());
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}

int main()
{
    std::cout << "Program started." << std::endl;
    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());

    // 定义范围为1000到2000的均匀分布
    std::uniform_int_distribution<> distrib(1000, 2048);
    // 打开CSV文件用于写入
    std::ofstream outFile("results.csv");
    // 写入标题行
    if (outFile.is_open())
    {
        outFile << "M,CPU-Time(ms),GPU-Time(ms)\n";
    }

    const int numTrials = 1000; // 设置重复执行的次数

    std::vector<double> matrix, vector, result;

    for (int M = 1; M <= 8; M++)
    {
        double totalGPUTime = 0;
        double totalCPUTime = 0;
        for (int trial = 0; trial < numTrials; ++trial)
        {
            int N = distrib(gen);
            int K = distrib(gen);
            init_matrix(matrix, N, K);
            init_vector(vector, K);
            init_result(result, N);
            float gputime = 0;
            float cputime = 0;
            processMatrixInStreams(matrix, vector, result, M, N, K, cputime, gputime);
            totalGPUTime += gputime;
            totalCPUTime += cputime;
        }
        float sum = 0;
        for (const auto element : result)
        {
            sum += element;
        }

        std::cout << "M: " << M << "\t"
                  << "CPU-time: " << totalCPUTime / numTrials << "\t"
                  << "GPU-time: " << totalGPUTime / numTrials << std::endl;

        std::cout << "sum: " << sum << std::endl;
        // 将M和time写入到文件
        outFile << M << "," << totalCPUTime / numTrials << "," << totalGPUTime / numTrials << "\n";
    }

    // 关闭文件
    outFile.close();

    std::cout << "Program ended." << std::endl;
    return 0;
}
