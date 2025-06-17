#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaStreamSynchronize) {
    cudaError cuda_retval;

    // create a new stream
    cudaStream_t new_stream = 0;

    // sync a new stream
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaStreamSynchronize, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &new_stream, .size = sizeof(cudaStream_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
} 