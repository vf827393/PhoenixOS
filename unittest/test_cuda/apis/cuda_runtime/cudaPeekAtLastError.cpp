#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaPeekAtLastError) {
    cudaError cuda_retval;

    // 检查初始状态（应该没有错误）
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaPeekAtLastError, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {}
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // // 触发一个错误（使用无效的事件句柄）
    // cuda_retval = (cudaError)this->_ws->pos_process(
    //     PosApiIndex_cudaEventSynchronize,
    //     this->_clnt->id,
    //     sync_params
    // );
    // EXPECT_EQ(cudaErrorInvalidResourceHandle, cuda_retval);

    // // 2. 检查是否仍保留错误状态
    // cuda_retval = (cudaError)this->_ws->pos_process(
    //     PosApiIndex_cudaPeekAtLastError,
    //     this->_clnt->id,
    //     {}
    // );
    // EXPECT_EQ(cudaErrorInvalidResourceHandle, cuda_retval);
} 