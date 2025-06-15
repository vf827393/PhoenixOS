#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetDeviceCount) {
    cudaError cuda_retval;
    int device_count;

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

    // // 测试空指针参数
    // POSAPIParamDesp null_param = { .value = nullptr, .size = sizeof(int) };
    // std::vector<POSAPIParamDesp> null_params = { null_param };
    
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaGetDeviceCount, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ null_params
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);
} 