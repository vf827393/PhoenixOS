#include"test_cuda/test_cuda_common.h"

__global__ void dummyKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = idx;
}

TEST_F(PhOSCudaTest, cudaFuncGetAttributes) {
    cudaFunctionAttributes attr;
    void *func_ptr = (void*)dummyKernel;

    cudaError cuda_retval = (cudaError)this->_ws->pos_process(
        /* api_id */ PosApiIndex_cudaFuncGetAttributes,
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &attr, .size = sizeof(cudaFunctionAttributes) },
            { .value = &func_ptr, .size = sizeof(void*) }
        }
    );

    EXPECT_EQ(cudaSuccess, cuda_retval);

    std::cout << "sharedSizeBytes: " << attr.sharedSizeBytes << std::endl;
    std::cout << "numRegs: " << attr.numRegs << std::endl;
    std::cout << "maxThreadsPerBlock: " << attr.maxThreadsPerBlock << std::endl;


    EXPECT_GT(attr.maxThreadsPerBlock, 0);
    EXPECT_GE(attr.numRegs, 0);
}
