#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

//Keeping track of input parameters
//Shape of Batch [10000]Shape of M (op ft maps) [12]Shape of C (ip ft maps) [1]Shape of Height [72]Shape of Kernel [7]Shape of Width [72]Shape of H_out [66]Shape of W_out [66]
//Shape of Batch [10000]Shape of M (op ft maps) [24]Shape of C (ip ft maps) [12]Shape of Height [33]Shape of Kernel [7]Shape of Width [33]Shape of H_out [27]Shape of W_out [27]
#define TILE_WIDTH_M1 16 //aligns well with warp notion even though 12 makes sense (16,16) tiles fits nicely for 32 threads.
#define TILE_WIDTH_M2 24 //aligns well with 2nd M value which is 24


//Do the unrolling when loading the values in shared memory by indexing and not by explicitely doing a unroll kernel and having additional memory constraints for storing that 
//which increases with K*K refer One disadvantage of forming the expanded input feature map matrix is that

//Ref page 385 version 4 of book
//it involves duplicating the input data up to KxK times, which can require the
//allocation of a prohibitively large amount of memory.

//https://lumetta.web.engr.illinois.edu/408-S19/slide-copies/ece408-lecture12-S19-ZJUI.pdf
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// __restrict__ doesnt do much change to performance but since we are only accessing pointers to index maybe it doesn't do any harm and might depend on compiler for optimization?
__global__ void Matmul_tiled_1(const float * __restrict__ Kernel, const float * __restrict__ X, float *  __restrict__ Y, int M, int C, int H, int W, int K, int H_out, int W_out) 
{   


  int numAColumns = K*K;
  int numCColumns = H_out*W_out;
  int b = blockIdx.z;
  X += b*H*W;
  Y += b*M*numCColumns;
  
  __shared__ float tileA[TILE_WIDTH_M1][TILE_WIDTH_M1];
  __shared__ float tileB[TILE_WIDTH_M1][TILE_WIDTH_M1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int colIx = blockIdx.x * TILE_WIDTH_M1 + tx;

  float result = 0;
  int q, p, w, h;

  //compiler directive to loop unroll to avoid processing load 
  #pragma unroll
  for (int tileIx = 0; tileIx < 4; ++tileIx) {

    int temp       = tileIx*TILE_WIDTH_M1;
    int matrix_col = temp + tx;

    if (ty < M && matrix_col < numAColumns)
      tileA[ty][tx] = Kernel[ty*numAColumns + matrix_col];  
    else
      tileA[ty][tx] = 0;

    int matrix_row = temp + ty;
    if (colIx < numCColumns && matrix_row < numAColumns) {
      q = matrix_row % K;
      matrix_row /= K;
      p = matrix_row % K;
      w = colIx % W_out;
      h = colIx / W_out;
      tileB[ty][tx] = X[(h+p) * (W) + w+q];
    }
    else 
      tileB[ty][tx] = 0;

    __syncthreads();

    #pragma unroll  
    for (int k = 0; k < TILE_WIDTH_M1; k++)
       result += tileA[ty][k]*tileB[k][tx];

    __syncthreads();   
  }
 
  if ((ty < M) && (colIx < numCColumns)) {
    Y[ty*numCColumns+colIx] = result;
  }
}

__global__ void Matmul_tiled_2(const float * __restrict__ Kernel, const float * __restrict__ X, float * __restrict__ Y, int M, int C, int H, int W, int K, int H_out, int W_out)
{

  int numAColumns = C*K*K;
  int numCColumns = H_out*W_out;
  int b = blockIdx.z;

  X += b*C*H*W;
  Y += b*M*numCColumns;

  __shared__ float tileA[TILE_WIDTH_M2][TILE_WIDTH_M2];
  __shared__ float tileB[TILE_WIDTH_M2][TILE_WIDTH_M2];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int colIx = blockIdx.x * TILE_WIDTH_M2 + tx;

  float result = 0;
  int q, p, c, w, h;

  #pragma unroll
  for (int tileIx = 0; tileIx < 25; ++tileIx){

    int temp       = tileIx*TILE_WIDTH_M2;
    int matrix_col = temp + tx;
    if (matrix_col < numAColumns)
      tileA[ty][tx] = Kernel[ty*numAColumns+matrix_col];
    else
      tileA[ty][tx] = 0;

    int matrix_row = temp + ty;
    if (matrix_row < numAColumns) {
      q = matrix_row % K;
      matrix_row /= K;
      p = matrix_row % K;
      c = matrix_row / K;
      w = colIx % W_out;
      h = colIx / W_out;
      tileB[ty][tx] = X[c * (H * W) + (h+p) * (W) + w+q];
    }
    else
      tileB[ty][tx] = 0;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_WIDTH_M2; k++)
       result += tileA[ty][k]*tileB[k][tx];

    __syncthreads();
  }

  if (colIx < numCColumns) {
    Y[ty*numCColumns+colIx] = result;
  }
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = H - K + 1;
    
    //use tile_width 24 for 2nd conv call
    if (M / 24){
        int gridDimY = ceil(1.0*M/TILE_WIDTH_M2), gridDimX = ceil(1.0*(H_out)*(W_out)/TILE_WIDTH_M2);
        dim3 gridDim (gridDimX, gridDimY, B), blockDim (TILE_WIDTH_M2, TILE_WIDTH_M2, 1);
        Matmul_tiled_2<<<gridDim, blockDim>>>(k.dptr_, x.dptr_, y.dptr_, M, C, H, W, K, H_out, W_out);
        // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }

    //use tile_Width 16 for 1st conv call
    else {
        int gridDimY = ceil(1.0*M/TILE_WIDTH_M1), gridDimX = ceil(1.0*(H_out)*(W_out)/TILE_WIDTH_M1);
        dim3 gridDim (gridDimX, gridDimY, B), blockDim (TILE_WIDTH_M1, TILE_WIDTH_M1, 1);
        Matmul_tiled_1<<<gridDim, blockDim>>>(k.dptr_, x.dptr_, y.dptr_, M, C, H, W, K, H_out, W_out);
        // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif

