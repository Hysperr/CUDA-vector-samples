#include <cuda.h>

__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(float *A, float *B, float *C, int n) {
    int size = n * sizeof(float);                       // bytes size of the vector
    float *d_A, *d_B, *d_C;

    cudaError_t err = cudaMalloc((void**) &d_A, size);  // allocate device memory, error checking
    if (err != cudaSuccess) {
        printf("%s in file %s on line %d\n", cudaGerErrorString(err), __FILE_, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);   // copy data

    cudaMalloc((void**) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_C, size);

    dim3 dimGrid(ceil(n / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);

    // kernel invocation, 1D grid
    vecAddKernel <<< ceil(n / 256.0), 256 >>> (d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

}
int main(int argc, char **argv) {
    int length = 10;
    float *h_A = malloc(sizeof(float) * length);
    float *h_B = malloc(sizeof(float) * length);
    float *h_C = malloc(sizeof(float) * length);

    for (int i = 0; i < length; i++) { h_A[i] = i; h_B[i] = i; }

    vecAdd(h_A, h_B, h_C, length);

    int tmpi = length;
    while (tmpi-- > 0) {
        if (h_C[i] != 2*h_A[i]) {
            printf("Test failed\n");
            break;
        }
        if (tmpi == 1) printf("Test passed\n");
    }

    return 0;

}
// ---------------------------------------------------------------------------------
/*
Covering a 76x62 picture with 16x16 blocks generates 80x64 threads, 20 blocks
results in gridDim.x = 5
results in gridDim.y = 4
int Plane = blockDim.z * blockIdx.z + threadIdx.z;  // z plane global index
linear access to P given z-plane is P[Plane*m*n + Row*n+Col];
*/
dim3 dimGrid(ceil(n / 16.0), ceil(m / 16.0), 1);
dim3 dimBlock(16, 16, 1);
pictureKernel <<< dimGrid, dimBlock >>> (tmp1, tmp2, tmp3, tmp4);

__global__
void pictureKernel(float *d_Pin, float *d_Pout, int n, int m) {
    // Calculate the row # of the d_Pin and d_Pout element to process
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    // Calculate the col # of the d_Pin and d_Pout element to process
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread computes one element of d_Pout if in range
    if ((Row < m) && (Col < n)) {
        d_Pout[Row*n+Col] = 2*d_Pout[Row*n+Col];
    }
}
// ---------------------------------------------------------------------------------
#define BLOCK_WIDTH 16
// Setup the execution configuration
int NumBlocks = Width / BLOCK_WIDTH;
if (width % BLOCK_WIDTH) NumBlocks++;
dim3 dimGrid(NumBlocks, Numblocks);
dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

__global__
matrixMultKernel(float *d_M, float *d_N, float *d_P, int width) {
    // Calculate the row index of the d_P element and d_M
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    // Calculate the col index of the d_P element and d_N
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((Row < width) && (Col < width)) {
        float Pvalue = 0;
        // Each thread computes one element of the block sub-matrix
        for (int k = 0; k < width; ++k) {
            Pvalue += d_M[Row * width + k] * [k * width + Col];
        }
        d_P[Row * width + Col] = Pvalue;
    }
}
// ---------------------------------------------------------------------------------
// querying device properties
int dev_count;
cudaGetDeviceCount(&dev_count);
cudaDeviceProp dev_prop;
for (int i = 0; i < dev_count; i++) {
cudaGetDeviceProperties(&dev_prop, i);
// decide if device has sufficient resources and capabilities
dev_prop.maxThreadsPerBlock;
dev_prop.multiProcessorCount;
dev_prop.clockRate;
dev_prop.warpSize;

dev_prop.maxThreadsDim[0];  // dimX
dev_prop.maxThreadsDim[1];  // dimY
dev_prop.maxThreadsDim[2];  // dimZ

dev_prop.maxGridSize[0];    // dimX
dev_prop.maxGridSize[1];    // dimY
dev_prop.maxGridSize[2];    // dimZ

}
// execution barrier synchronization function
__syncthreads()
// ---------------------------------------------------------------------------------
// 1D basic convolution

#define MAX_MASK_WIDTH 10
__constant__ float M[MAX_MASK_WIDTH]  // constant global var, no need to pass in M to kernel now since in global memory

__global__
void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width / 2);
    for (int j = 0; j < Mask_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
}
// ---------------------------------------------------------------------------------
/* Matrix Mult using tiling. Reducing Global Memory (DRAM) Accesses
   Using shared memory, increased CGMA ratio
*/
# define TILE_WIDTH 16

__global__
void MatrixMulKernel(float* d_M, float *d_N, float *d_P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    // Loop over the d_M and d_N tiles required to compute d_P element
    for (int m = 0; m < Width / TILE_WIDTH; ++m) {

        // Collaborative loading of d_M and d_N tiles into shared memory
        Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    d_P[Row * Width + Col] = Pvalue;
}