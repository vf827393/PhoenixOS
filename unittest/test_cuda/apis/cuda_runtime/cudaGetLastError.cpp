#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetLastError) {
    cudaError cuda_retval;

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetLastError, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {}
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // cudaEvent_t invalid_event = nullptr;
    // POSAPIParamDesp sync_param = { .value = &invalid_event, .size = sizeof(cudaEvent_t) };
    // std::vector<POSAPIParamDesp> sync_params = {sync_param};
    
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaStreamSynchronize, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ sync_params
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);

    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaGetLastError, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {}
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);


    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaGetLastError, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {}
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);
} 