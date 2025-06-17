#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetErrorString) {
    cudaError cuda_retval;
    cudaError_t error_string = cudaSuccess;

    // test scuessful state error string
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetErrorString, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &error_string, .size = sizeof(cudaError_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    //EXPECT_STREQ("no error", error_string);

//     cudaError_t invalid_error = cudaErrorInvalidValue;
//     POSAPIParamDesp invalid_param1 = { .value = &invalid_error, .size = sizeof(cudaError_t) };
//     POSAPIParamDesp invalid_param2 = { .value = &error_string, .size = sizeof(const char*) };
//     std::vector<POSAPIParamDesp> invalid_params = {invalid_param1, invalid_param2};
    
//     cuda_retval = (cudaError)this->_ws->pos_process( 
//         /* api_id */ PosApiIndex_cudaGetErrorString, 
//         /* uuid */ this->_clnt->id,
//         /* param_desps */ invalid_params
//     );
//     EXPECT_EQ(cudaSuccess, cuda_retval);
//     EXPECT_STREQ("invalid argument", error_string);

//     cudaError_t memory_error = cudaErrorMemoryAllocation;
//     POSAPIParamDesp memory_param1 = { .value = &memory_error, .size = sizeof(cudaError_t) };
//     POSAPIParamDesp memory_param2 = { .value = &error_string, .size = sizeof(const char*) };
//     std::vector<POSAPIParamDesp> memory_params = {memory_param1, memory_param2};
    
//     cuda_retval = (cudaError)this->_ws->pos_process( 
//         /* api_id */ PosApiIndex_cudaGetErrorString, 
//         /* uuid */ this->_clnt->id,
//         /* param_desps */ memory_params
//     );
//     EXPECT_EQ(cudaSuccess, cuda_retval);
//     EXPECT_STREQ("out of memory", error_string);
} 