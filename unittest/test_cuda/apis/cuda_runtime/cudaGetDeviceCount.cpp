#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetDeviceCount) {
    cudaError cuda_retval;
    int device_count;
    int *device_count_ptr = &device_count;
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDeviceCount, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &device_count_ptr, .size = sizeof(int*) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    EXPECT_GT(device_count, 0);
}
