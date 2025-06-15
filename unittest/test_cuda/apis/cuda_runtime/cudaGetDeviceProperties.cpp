#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaGetDeviceProperties) {
    cudaError cuda_retval;
    int device_count;
    cudaDeviceProp prop;

    // 获取设备数量
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDeviceCount, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &device_count, .size = sizeof(int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    EXPECT_GT(device_count, 0);

    // 获取当前设备的属性
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDeviceProperties, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
             {.value = &prop, .size = sizeof(cudaDeviceProp)} 
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // 验证一些基本属性
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

    // 测试无效设备号
    int invalid_device = device_count;
    POSAPIParamDesp invalid_param1 = { .value = &invalid_device, .size = sizeof(int) };
    POSAPIParamDesp invalid_param2 = { .value = &prop, .size = sizeof(cudaDeviceProp) };
    std::vector<POSAPIParamDesp> invalid_params = {invalid_param1, invalid_param2};
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaGetDeviceProperties, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ invalid_params
    );
    EXPECT_EQ(cudaErrorInvalidDevice, cuda_retval);
} 