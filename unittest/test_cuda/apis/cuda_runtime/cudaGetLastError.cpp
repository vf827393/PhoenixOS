#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetLastError) {
    cudaError cuda_retval;

    // 检查初始状态（应该没有错误）
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetLastError, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {}
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // // 触发一个错误（使用无效的事件句柄）
    // cudaEvent_t invalid_event = nullptr;
    // POSAPIParamDesp sync_param = { .value = &invalid_event, .size = sizeof(cudaEvent_t) };
    // std::vector<POSAPIParamDesp> sync_params = {sync_param};
    
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaStreamSynchronize, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ sync_params
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);

    // // 获取错误状态（应该返回错误）
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaGetLastError, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {}
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);

    // // 再次获取错误状态（应该返回 cudaSuccess，因为错误已经被清除）
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaGetLastError, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {}
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);
} 