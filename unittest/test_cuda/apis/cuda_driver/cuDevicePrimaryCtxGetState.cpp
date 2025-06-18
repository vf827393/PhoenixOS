#include "test_cuda/test_cuda_common.h"


TEST_F(PhOSCudaTest, cuDevicePrimaryCtxGetState) {
    CUresult cuda_retval;
    int dev = 0;
    CUdevice device;
    CUdevice* device_ptr = &device;
    unsigned int flags;
    unsigned int* flags_ptr = &flags;
    int active;
    int *active_ptr = &active;

    cuda_retval = (CUresult)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuDeviceGet, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &device_ptr, .size = sizeof(CUdevice*) },
            { .value = &dev, .size = sizeof(int) }
        }
    );
    EXPECT_EQ(CUDA_SUCCESS, cuda_retval);

    cuda_retval = (CUresult)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuDevicePrimaryCtxGetState, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &device, .size = sizeof(CUdevice) },
            { .value = &flags_ptr, .size = sizeof(unsigned int*) },
            { .value = &active_ptr, .size = sizeof(int*) }
        }
    );
    EXPECT_EQ(CUDA_SUCCESS, cuda_retval);
}
