#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetDeviceProperties) {
    cudaError cuda_retval;
    int i, device_count;
    int *device_count_ptr = &device_count;
    cudaDeviceProp prop;
    cudaDeviceProp *prop_ptr = &prop;

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDeviceCount, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &device_count_ptr, .size = sizeof(int*) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    EXPECT_GT(device_count, 0);

    for(i=0; i<device_count; i++){
        memset(prop_ptr, 0, sizeof(cudaDeviceProp));
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaGetDeviceProperties, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                {.value = &prop_ptr, .size = sizeof(cudaDeviceProp*)},
                {.value = &i, .size = sizeof(int)}
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
        EXPECT_GT(prop.major, 0);
        EXPECT_GE(prop.minor, 0);
        EXPECT_GT(prop.totalGlobalMem, 0);
        EXPECT_GT(prop.multiProcessorCount, 0);
        EXPECT_GT(prop.maxThreadsPerBlock, 0);
        EXPECT_GT(prop.maxThreadsPerMultiProcessor, 0);
        EXPECT_GT(prop.maxBlocksPerMultiProcessor, 0);
        EXPECT_GT(prop.maxGridSize[0], 0);
        EXPECT_GT(prop.maxGridSize[1], 0);
        EXPECT_GT(prop.maxGridSize[2], 0);
        EXPECT_GT(prop.maxThreadsDim[0], 0);
        EXPECT_GT(prop.maxThreadsDim[1], 0);
        EXPECT_GT(prop.maxThreadsDim[2], 0);
    }
}
