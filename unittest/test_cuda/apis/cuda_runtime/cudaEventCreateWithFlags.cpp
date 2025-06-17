#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaEventCreateWithFlags) {
    cudaError cuda_retval;
    cudaEvent_t event;
    cudaEvent_t *event_ptr = &event;
    unsigned int flags = 0; 
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event_ptr, .size = sizeof(cudaEvent_t*) },
            { .value = &flags, .size = sizeof(unsigned int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
}
