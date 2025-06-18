#include "test_cuda/test_cuda_common.h"


TEST_F(PhOSCudaTest, cuDeviceGet) {
    CUresult cuda_retval;
    int dev = 0;
    CUdevice device;
    CUdevice* device_ptr = &device;

    cuda_retval = (CUresult)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuDeviceGet, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &device_ptr, .size = sizeof(CUdevice*) },
            { .value = &dev, .size = sizeof(int) }
        }
    );
    EXPECT_EQ(CUDA_SUCCESS, cuda_retval);
}
