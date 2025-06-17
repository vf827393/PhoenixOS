#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaEventDestroy) {
    cudaError cuda_retval;
    cudaEvent_t event;

    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventDestroy, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);


    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventDestroy, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventDestroy, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);
} 