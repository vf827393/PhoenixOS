#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaEventQuery) {
    cudaError cuda_retval;
    cudaEvent_t event ;
    cudaStream_t stream = 0;

    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);


    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventRecord, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) },
            { .value = &stream, .size = sizeof(cudaStream_t) },
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);


    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );

    EXPECT_TRUE(cuda_retval == cudaSuccess || cuda_retval == cudaErrorNotReady);


    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
             { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
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
} 