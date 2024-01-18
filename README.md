# EECS 471 Final Project

Private repo for EECS-471 Applied GPU Programming FInal Project. Goal is to perform optimization in the forward convolution pass using the MXNet Library and CUDA programming.

## Introduction

The goal of the project is the following:

* Get practical experience by using, profiling, and modifying MXNet, a standard open-source neural-network framework.
* Demonstrate command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolution layer forward pass.

## Strategies Considered

* ✅ Tuning with restrict and loop unrolling.
* ✅ Sweeping various parameters to find best values (block sizes, amount of thread coarsening)
* ✅ Exploiting parallelism in input images, input channels, and output channels.
* ✅ Multiple kernel implementations for different layer sizes

## Optimizations 

### ✅ [Optimization One: Constant Memory with inner for loop](ece408_src/new-forward_4_unroll.cuh)
* The first optimization was made by moving the filter bank W to constant memory. Since constant memory
allows for near instantaneous access to data and since the filter bank remained constant throughout the
execution of the kernel, we correctly assumed that moving W to constant memory would result in a
sizable speed increase. The constant memory was uploaded by the host and simply read every time the
device needed to find the convolution layer. When we tried this change, the final project was sped up to a
mere .14 seconds. This is logical as the we were able to eliminate B*M*W*H*C*K^2 calls to global
memory as shown in the implementation below. (Each individual thread read global memory C*K^2
times and there were W*H*B*M*Z threads used).

### ✅ [Optimization Two: Shared Memory convolution](ece408_src/new-forward_4_sharedMemory.cuh)
* The configuration involves organizing blocks into a 3D grid structure to execute a convolutional neural network's forward pass. The second optimization implemented was utilizing shared memory for the convolution layer as described
in chapter 16 of the textbook. This was implemented due to the recommendation that it would
substantially reduce memory bandwidth which was the primary bottleneck in the starter code. The way this worked was that first, the filter bank W was stored into shared memory with each thread computing one value in the array. Second, the required portion of the input X to compute the output tile was added to another shared array. Finally, the partial sum of Y was computed.

### ✅ [Optimization Three: Tiled Matrix Multiplication with different Tilesizes based on OP feature maps for 2 different conv layers](ece408_src/new-forward_4_shared_constMem.cuh)

* This optimization is by far the most results-fetching one which we’ve implemented. The Fig 16.17 in 4th edition of the book “Programming Massively Parallel Processes” gives us with a GEMM approach by unrolling the kernel X using a separate kernel X_unroll (growth of input feature map by K*K times) for having a nice and single large matrix multiplication between the filters W and X_unrolled to get the corresponding convolution output. However the problem with this approach is 2 fold, one you are spending extra memory and global memory read for the unroll kernel and second you spend time again to do the logic for this single large matrix multiplication once you generate the values for X_unroll using proper indexing. In our approach we use the fact that if we are already doing a single large matrix multiplication why not make use of shared memory and do tile matrix multiplication, while leveraging the fact that unroll the appropriate elements right at the time of loading the elements in Tile by proper indexing. Thereby we avoid the need to have a separate kernel to do the unrolling while simultaneously reducing the redundant loading and storing which has to be done if the kernel described in the book is to be used.

* Another important optimization which we did was use two different tile sizes for both the convolution forward passes. 


### Installing CUDA and usage

The model runs on the latest version of cuda with Nsight enabled. The entirety of the code 
was implemented using greatlakes cluster.

You can download the CUDA toolkit from: https://developer.nvidia.com/cuda-downloads. 
Follow the installation instructions. 
