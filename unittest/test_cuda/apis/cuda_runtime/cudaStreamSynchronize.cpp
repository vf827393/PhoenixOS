#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaStreamSynchronize) {
    cudaError cuda_retval;

    // 创建一个新的流
    cudaStream_t new_stream = 0;

    // 同步新创建的流
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaStreamSynchronize, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &new_stream, .size = sizeof(cudaStream_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
} 