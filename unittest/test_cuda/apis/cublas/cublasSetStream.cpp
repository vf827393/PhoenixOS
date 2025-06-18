#include "test_cuda/test_cuda_common.h"


TEST_F(PhOSCudaTest, cublasSetStream) {
    cublasStatus_t cublas_retval;
    cublasHandle_t cublas_context;
    cublasHandle_t *cublas_context_ptr = &cublas_context;
    cudaStream_t stream = 0;

    cublas_retval = (cublasStatus_t)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cublasCreate_v2, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &cublas_context_ptr, .size = sizeof(cublasHandle_t *) }
        }
    );
    EXPECT_EQ(CUBLAS_STATUS_SUCCESS, cublas_retval);

    cublas_retval = (cublasStatus_t)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cublasSetStream, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &cublas_context, .size = sizeof(cublasHandle_t) },
            {.value = &stream, .size = sizeof(cudaStream_t) }
        }
    );
    EXPECT_EQ(CUBLAS_STATUS_SUCCESS, cublas_retval);
}
