#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetDevice) {
    cudaError cuda_retval;
    int device = -1;

    // 获取当前设备
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDevice, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &device, .size = sizeof(int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    EXPECT_GE(device, 0);
    //EXPECT_GE(device, 0);

    // // 测试空指针参数
    // POSAPIParamDesp null_param = { .value = nullptr, .size = sizeof(int) };
    // std::vector<POSAPIParamDesp> null_params = { null_param };
    
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaGetDevice, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ null_params
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);
} 