#include "test_cuda/test_cuda_common.h"


TEST_F(PhOSCudaTest, cublasCreatev2) {
    cublasStatus_t cublas_retval;
    cublasHandle_t cublas_context;
    cublasHandle_t *cublas_context_ptr = &cublas_context;

    cublas_retval = (cublasStatus_t)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cublasCreate_v2, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &cublas_context_ptr, .size = sizeof(cublasHandle_t *) }
        }
    );
    EXPECT_EQ(CUBLAS_STATUS_SUCCESS, cublas_retval);
}
