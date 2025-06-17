#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaStreamIsCapturing) {
    cudaError cuda_retval;
    cudaStream_t stream = 0;  
    cudaStreamCaptureStatus status;
    cudaStreamCaptureStatus *status_ptr = &status;

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaStreamIsCapturing, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
                { .value = &stream, .size = sizeof(cudaStream_t) },
                { .value = &status_ptr, .size = sizeof(cudaStreamCaptureStatus*) },
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
}
