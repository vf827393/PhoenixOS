#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetErrorString) {
    cudaError cuda_retval;
    cudaError_t error_string = cudaSuccess;

    // 测试成功状态的错误字符串
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetErrorString, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            {.value = &error_string, .size = sizeof(cudaError_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    //EXPECT_STREQ("no error", error_string);

//     // 测试无效值的错误字符串
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

//     // 测试内存分配错误的错误字符串
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