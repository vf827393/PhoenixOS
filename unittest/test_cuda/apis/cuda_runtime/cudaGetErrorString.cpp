#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetErrorString) {
    cudaError cuda_retval;
    cudaError_t error_string = cudaSuccess;

    // test scuessful state error string
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaGetErrorString, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ { 
    //         {.value = &error_string, .size = sizeof(cudaError_t) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);
    //EXPECT_STREQ("no error", error_string);
}
