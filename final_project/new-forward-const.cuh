
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 32

namespace mxnet
{
namespace op
{

__constant__ float constMem[8192];

//Optimization 1 : ---------Shared memory convolution + weights in constant cache----------//

__global__ void forward_kernel_shared_const(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


    //General formula to calculate output is [(W-K+2P)/S] + 1 
    //In this case since alex net doesnt use padding and stride is always 1,
    //the following H_out , W_out calculation can be used {P=0, S=1}.


    //input feature map X[B,C,H,W]
    //output feature map Y[B,M,H-K+1,W-K+1], where H and W are height and width of 
    //each input map image, there are M x C filter banks in total which are stored in 
    //W of shape W[M,C,K,K]. 


    const int H_out = H - K + 1; 
    const int W_out = W - K + 1;
    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int H_grid = ceil((float)H_out/TILE_WIDTH);

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    //Kernel 1 :---------Shared memory convolution-----------//

    int bz = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int TILE_WIDTH_K = TILE_WIDTH + K-1;


    extern __shared__ float shared[];

    //this kernel needs to load only the X elements to shared and W is already placed in constant mem cache
    float* sharedX = &shared[0];
    //float* sharedW = &shared[TILE_WIDTH_K*TILE_WIDTH_K];

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h_base = (bz / W_grid) * TILE_WIDTH;
    int w_base = (bz % W_grid) * TILE_WIDTH;
    int h = h_base + ty;
    int w = w_base + tx;

    float acc = 0.0;
    float weight;

    // 1. load X into the shared memory  must include additional loop for inputs channel map
    for (int c=0 ; c<C; c++)
    {
      for (int i=h; i<h_base+TILE_WIDTH_K; i+=TILE_WIDTH) 
      {
        for (int j=w; j<w_base+TILE_WIDTH_K; j+=TILE_WIDTH) 
        {
          if (b<B && i<H && j<W) 
          {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = x4d(b,c, i, j);
          } else 
          {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = 0;
          }
          
        }
      }
      __syncthreads();

      // 2. compute partial sum of output Y
      for (int p=0; p<K; p++) 
      {
        for (int q=0; q<K; q++) 
        {
          if (((ty+p)<TILE_WIDTH_K) && ((tx+q)<TILE_WIDTH_K)) 
          {
            weight = constMem[m*C*K*K+ c*K*K+p*K+q]; //need to verify this linearized indexed load from constant cache
            acc += sharedX[(ty+p)*TILE_WIDTH_K+(tx+q)]*weight;
          }
        }
      }
      __syncthreads();
    }

    if (b<B && m<M && h<H_out && w<W_out) {
      y4d(b, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    cudaStream_t s = y.stream_->stream_;

    const int B = x.shape_[0]; // num of batch
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1]; // num of channel
    const int H = x.shape_[2]; // height
    const int W = x.shape_[3]; // width
    const int K = w.shape_[3]; // size of kernel or filter

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int TILE_WIDTH_K = TILE_WIDTH + K - 1;
    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int Z = H_grid * W_grid;

    // ===== Optimization 2: Shared Memory Convolution + constant cache for W filters==== 
    // Set the kernel dimensions

    //copy the W to constant memory
    cudaMemcpyToSymbol(constMem,w.dptr_,M*C*K*K*sizeof(float));
    
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    size_t shared = sizeof(float)*(TILE_WIDTH_K*TILE_WIDTH_K+K*K);
    forward_kernel_shared_const<<<gridDim, blockDim, shared, s>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed for ECE408");
}
}
}

#endif

