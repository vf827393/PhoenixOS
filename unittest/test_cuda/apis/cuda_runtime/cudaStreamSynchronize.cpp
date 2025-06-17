#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaStreamSynchronize) {
    cudaError cuda_retval;
    cudaStream_t stream = 0;

    // sync stream
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaStreamSynchronize, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &stream, .size = sizeof(cudaStream_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
}
