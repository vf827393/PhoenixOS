#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaPeekAtLastError) {
    cudaError cuda_retval;

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaPeekAtLastError, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {}
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // cuda_retval = (cudaError)this->_ws->pos_process(
    //     PosApiIndex_cudaEventSynchronize,
    //     this->_clnt->id,
    //     sync_params
    // );
    // EXPECT_EQ(cudaErrorInvalidResourceHandle, cuda_retval);


    // cuda_retval = (cudaError)this->_ws->pos_process(
    //     PosApiIndex_cudaPeekAtLastError,
    //     this->_clnt->id,
    //     {}
    // );
    // EXPECT_EQ(cudaErrorInvalidResourceHandle, cuda_retval);
} 