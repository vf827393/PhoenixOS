#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaEventRecordWithFlags) {
    cudaError cuda_retval;
    cudaEvent_t event ;
    cudaStream_t stream = 0;
    unsigned int flags = 0;  // 使用阻塞同步标志

    // 创建事件
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventCreate, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // 记录事件（带标志）

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventRecordWithFlags, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) },
            { .value = &stream, .size = sizeof(cudaStream_t) },
            { .value = &flags, .size = sizeof(unsigned int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // 销毁事件

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventDestroy, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);
} 