#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetDevice) {
    cudaError cuda_retval;
    int device = -1;
    int* device_ptr = &device;

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDevice, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &device_ptr, .size = sizeof(int*) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    EXPECT_GE(device, 0);
}
