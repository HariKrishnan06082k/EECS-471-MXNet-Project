
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16

namespace mxnet
{
namespace op
{


//Optimization 1 : ---------Shared memory convolution----------//

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
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

    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    //Kernel 1 :---------Shared memory convolution-----------//

    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int W_grid = ceil((float)W_out/TILE_WIDTH);

    int TILE_WIDTH_K = TILE_WIDTH + K -1;
    extern __shared__ float shared[];
    float* sharedX = &shared[0];
    float* sharedW = &shared[TILE_WIDTH_K*TILE_WIDTH_K];

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h_base = (bz / W_grid) * TILE_WIDTH;
    int w_base = (bz % W_grid) * TILE_WIDTH;
    int h = h_base + ty;
    int w = w_base + tx;

    float acc = 0.0;

    for (int c=0; c<C; c++) {
      // 1. load the filter bank W in the shared memory
      if ((tx < K) && (ty < K)) {
        sharedW[ty*K+tx] = k4d(m, c, ty, tx);
      }
      __syncthreads();

      // 2. load tile from X into the shared memory
      for (int i=h; i<h_base+TILE_WIDTH_K; i+=TILE_WIDTH) {
        for (int j=w; j<w_base+TILE_WIDTH_K; j+=TILE_WIDTH) {
          if (i<H && j<W) {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = x4d(b,c, i, j);
          } else {
            sharedX[(i-h_base)*TILE_WIDTH_K+(j-w_base)] = 0;
          }
          
        }
      }
      __syncthreads();

      // 3. compute partial sum of output Y
      for (int p=0; p<K; p++) {
        for (int q=0; q<K; q++) {
          if (((ty+p)<TILE_WIDTH_K) && ((tx+q)<TILE_WIDTH_K)) {
            acc += sharedX[(ty+p)*TILE_WIDTH_K+(tx+q)]*sharedW[p*K+q];
          }
        }
      }
      __syncthreads();
    }

    if (b<B && m<M && h<H_out && w<W_out) {
      y4d(b, m, h, w) = acc;
    }


//This is the CPU implementation don't use it.
/*
    int b = blockDim.x * blockIdx.x + threadIdx.x;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++)
                {
                    y4d(b, m, h, w) = 0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                }
    }

*/
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

    const int B = x.shape_[0]; // num of batch
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1]; // num of channel
    const int H = x.shape_[2]; // height
    const int W = x.shape_[3]; // width
    const int K = w.shape_[3]; // size of kernel or filter

    dim3 gridDim((B + 511) / 512);
    dim3 blockDim(512);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    //put kernel call here 
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
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
