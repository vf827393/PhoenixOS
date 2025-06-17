#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaSetDevice) {
    cudaError cuda_retval;
    int i, device_count;
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

    for(i=0; i<device_count; i++){
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaSetDevice, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &i, .size = sizeof(int) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
    }
}
