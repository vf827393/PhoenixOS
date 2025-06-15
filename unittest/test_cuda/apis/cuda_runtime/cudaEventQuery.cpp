#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaEventQuery) {
    cudaError cuda_retval;
    cudaEvent_t event ;
    cudaStream_t stream = 0;

    // 创建事件
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // 记录事件

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventRecord, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) },
            { .value = &stream, .size = sizeof(cudaStream_t) },
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // 查询事件状态

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    // 由于事件刚刚被记录，可能还未完成，所以这里不检查具体返回值
    EXPECT_TRUE(cuda_retval == cudaSuccess || cuda_retval == cudaErrorNotReady);

    // 同步事件

    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
             { .value = &event, .size = sizeof(cudaEvent_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // 再次查询事件状态
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventQuery, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
             { .value = &event, .size = sizeof(cudaEvent_t) }
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