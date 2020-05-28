#include <iostream>
#include <algorithm>
#include <array>
#include <numeric>
#include <random>

#include <cooperative_groups.h>

#define N  65536
#define TB  32
#define GB  1000
#define NGPU 8
#define TYPE int

//using namespace cooperative_groups;
namespace cg = cooperative_groups;

template<typename T>
__device__ int reduce_thread(T x) {
    unsigned int mask;
    for (int i = warpSize/2; i > 0; i >>= 1) {
        mask = __activemask();
        x += __shfl_down_sync(mask, x, i);
    }
    return x;
}

template<class T>
__global__ void reduce_kernel(T* data,T* sum) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    T x = data[tid];

    cg::multi_grid_group multi_grid=cg::this_multi_grid();
    extern __shared__ T shared[];

    int wid=threadIdx.x/warpSize;

    x=reduce_thread(x); //implictly synchoriz

    if ((tid % warpSize) == 0)
        shared[wid] = x;
    //cg::this_thread_block().sync();

    if (threadIdx.x ==0) {
        T s=0;
        for (int i=0;i<blockDim.x/warpSize;i++) {
            //printf(">%d,%d\n",i,shared[i]);
            s += shared[i];
        }
        data[blockIdx.x]=s;
    }

    cg::this_grid().sync();
    multi_grid.sync();

    if (tid==0) {
        T s=0;
        for (int i=0;i<gridDim.x;i++) {
            s += data[i];
        }
        atomicAdd(sum,s);
    }
}

int main() {
    void *d_buf[NGPU];
    void* args[NGPU][2];
    TYPE *d_sum;

    std::array<TYPE, N> mydata;

    std::generate(mydata.begin(),mydata.end(),std::rand);
    int result=std::accumulate(mydata.begin(),mydata.end(),0);

    dim3 grid((N + TB-1) / TB / NGPU);
    dim3 blk(TB);

    int sum = 0;
    int cp_size;
    int saveDevice;

    cudaGetDevice(&saveDevice);
    for (auto i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cp_size = sizeof(TYPE) * mydata.size() / NGPU;
        cudaMalloc(&d_buf[i], cp_size);
        cudaMemcpy(d_buf[i], mydata.data()+(cp_size*i/sizeof(TYPE)), cp_size, cudaMemcpyHostToDevice);
        args[i][0]=&d_buf[i];
        args[i][1]=&d_sum;
    }
    cudaSetDevice(saveDevice);

    cudaMallocManaged(&d_sum, sizeof(TYPE));
    *d_sum=0;
    //cudaMemset(d_sum, 0, sizeof(int));

    int shareMem = sizeof(TYPE) * (TB / 32);

    cudaLaunchParams params[NGPU];

    cudaGetDevice(&saveDevice);

    for (auto i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        params[i].func = (void *) reduce_kernel<int>;
        params[i].blockDim = blk;
        params[i].gridDim = grid;
        params[i].args = args[i];
        params[i].sharedMem = shareMem;
        cudaStreamCreate(&params[i].stream);
    }
    cudaSetDevice(saveDevice);
    cudaError_t err = cudaLaunchCooperativeKernelMultiDevice(params, NGPU,0);

    if (err != cudaSuccess) {
        std::cout << "CUDA run failed: " << shareMem << " " << cudaGetErrorString(err) << std::endl;
    }
    for (auto i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(params[i].stream);
    }
    cudaDeviceSynchronize();
    std::cout<<"All data reduce with "<<NGPU<<" GPUs = "<<*d_sum;
    std::cout<<((result==*d_sum) ? ", PASSED!\n" : ", FAILED!\n");
    return 0;
}
