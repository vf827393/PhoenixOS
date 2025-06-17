#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaEventCreateWithFlags) {
    cudaError cuda_retval;
    cudaEvent_t event;
    unsigned int flags = 0; 
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) },
            { .value = &flags, .size = sizeof(unsigned int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // test different sign
    // flags = cudaEventBlockingSync;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);


    // flags = cudaEventDisableTiming;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);

    // flags = cudaEventDisableTiming | cudaEventInterprocess;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);

    // flags = cudaEventInterprocess;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);


    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventDestroy, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);
} 