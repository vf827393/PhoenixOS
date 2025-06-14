#include "test_cuda/test_cuda_common.h"


TEST_F(PhOSCudaTest, cudaDeviceGetAttribute) {
    cudaError cuda_retval;
    int i, count = 0, value;
    int *count_ptr = &count, *value_ptr = &value;
    cudaDeviceAttr attr = cudaDevAttrMaxThreadsPerBlock;
    
    // obtain device count
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDeviceCount, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &count_ptr, .size = sizeof(int*) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // obtain device attribute
    for(i=0; i<count; i++){
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaDeviceGetAttribute, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &value_ptr, .size = sizeof(int*) },
                { .value = &attr, .size = sizeof(cudaDeviceAttr) },
                { .value = &i, .size = sizeof(int) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
    }
}
