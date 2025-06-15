#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaStreamIsCapturing) {
    cudaError cuda_retval;
    cudaStream_t stream = 0;  // 使用默认流
    cudaStreamCaptureStatus **status;

    // 检查默认流的捕获状态
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaStreamIsCapturing, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
                { .value = &stream, .size = sizeof(cudaStream_t) },
                { .value = &status, .size = sizeof(cudaStreamCaptureStatus**) },
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
    // EXPECT_EQ(cudaStreamCaptureStatusNone, status);

    // // 创建一个新的流
    // cudaStream_t new_stream;
    // POSAPIParamDesp create_param = { .value = &new_stream, .size = sizeof(cudaStream_t) };
    // std::vector<POSAPIParamDesp> create_params = {create_param};
    
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreate, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ create_params
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);

    // // 检查新创建的流的捕获状态
    // check_params[0].value = &new_stream;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaStreamIsCapturing, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ check_params
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);
    // EXPECT_EQ(cudaStreamCaptureStatusNone, status);

    // // 销毁新创建的流
    // POSAPIParamDesp destroy_param = { .value = &new_stream, .size = sizeof(cudaStream_t) };
    // std::vector<POSAPIParamDesp> destroy_params = {destroy_param};

    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventDestroy, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ destroy_params
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);
} 