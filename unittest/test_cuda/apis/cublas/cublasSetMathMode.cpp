#include "test_cuda/test_cuda_common.h"


TEST_F(PhOSCudaTest, cublasSetMathMode) {
    cublasStatus_t cublas_retval;
    cublasHandle_t cublas_context;
    cublasHandle_t *cublas_context_ptr = &cublas_context;
    cublasMath_t mode = CUBLAS_DEFAULT_MATH;

    cublas_retval = (cublasStatus_t)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cublasCreate_v2, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &cublas_context_ptr, .size = sizeof(cublasHandle_t *) }
        }
    );
    EXPECT_EQ(CUBLAS_STATUS_SUCCESS, cublas_retval);

    cublas_retval = (cublasStatus_t)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cublasSetMathMode, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &cublas_context, .size = sizeof(cublasHandle_t) },
            {.value = &mode, .size = sizeof(cublasMath_t) }
        }
    );
    EXPECT_EQ(CUBLAS_STATUS_SUCCESS, cublas_retval);
}
