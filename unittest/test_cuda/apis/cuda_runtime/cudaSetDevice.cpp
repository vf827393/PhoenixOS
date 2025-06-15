#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaSetDevice) {
    cudaError cuda_retval;
    int device_count;
    int current_device;

    // 获取设备数量
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDeviceCount, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &device_count, .size = sizeof(int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    EXPECT_GT(device_count, 0);

    // 获取当前设备
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDevice, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &current_device, .size = sizeof(int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // 如果有多个设备，测试切换到其他设备
    if (device_count > 1) {
        int new_device = (current_device + 1) % device_count;
        
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaSetDevice, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &new_device, .size = sizeof(int) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
    }

} 