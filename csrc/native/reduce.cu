#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define WarpSize 32
//latency: 1.254ms
template <int blockSize>
__device__ float WarpShuffle(float sum) {
    // 使用模板参数保持与调用处的一致性
    // 优化shfl指令顺序，保持注释清晰度
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template <int blockSize>
__global__ void reduce_warp_level(float *d_in,float *d_out, unsigned int n){
    float sum = 0;//当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器

    unsigned int tid = threadIdx.x;
    unsigned int gtid = blockIdx.x * blockSize + threadIdx.x;
    // 分配的线程总数
    unsigned int total_thread_num = blockSize * gridDim.x;
    // 基于v5的改进：不用显式指定一个线程处理2个元素，而是通过L30的for循环来自动确定每个线程处理的元素个数
    for (int i = gtid; i < n; i += total_thread_num)
    {
        sum += d_in[i];
    }
    
    // 用于存储partial sums for each warp of a block
    __shared__ float WarpSums[blockSize / WarpSize]; 
    // 当前线程在其所在warp内的ID
    const int laneId = tid % WarpSize;
    // 当前线程所在warp在所有warp范围内的ID
    const int warpId = tid / WarpSize; 
    // 对当前线程所在warp作warpshuffle操作，直接交换warp内线程间的寄存器数据
    sum = WarpShuffle<blockSize>(sum);
    if(laneId == 0) {
        WarpSums[warpId] = sum;
    }
    __syncthreads();
    //至此，得到了每个warp的reduce sum结果
    //接下来，再使用第一个warp(laneId=0-31)对每个warp的reduce sum结果求和
    //首先，把warpsums存入前blockDim.x / WarpSize个线程的sum寄存器中
    //接着，继续warpshuffle
    sum = (tid < blockSize / WarpSize) ? WarpSums[tid] : 0;
    // Final reduce using first warp
    if (warpId == 0) {
        sum = WarpShuffle<blockSize/WarpSize>(sum); 
    }
    // store: 哪里来回哪里去，把reduce结果写回显存
    if (tid == 0) {
        d_out[blockIdx.x] = sum;
    }
}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_warp_level<blockSize><<<Grid,Block>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("resPerBlock : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_warp_level latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}