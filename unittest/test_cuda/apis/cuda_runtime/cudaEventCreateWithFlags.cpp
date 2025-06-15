#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaEventCreateWithFlags) {
    cudaError cuda_retval;
    cudaEvent_t event;
    unsigned int flags = 0; // 使用默认标志

    // 创建事件（带标志）
    
    cuda_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &event, .size = sizeof(cudaEvent_t) },
            { .value = &flags, .size = sizeof(unsigned int) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_retval);

    // // 测试不同的标志组合
    // // 1. 阻塞同步标志
    // flags = cudaEventBlockingSync;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);

    // // 2. 禁用计时标志
    // flags = cudaEventDisableTiming;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);

    // // 3. 进程间标志（需要同时设置禁用计时标志）
    // flags = cudaEventDisableTiming | cudaEventInterprocess;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);

    // // 4. 测试无效标志组合（进程间标志未设置禁用计时标志）
    // flags = cudaEventInterprocess;
    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventCreateWithFlags, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) },
    //         { .value = &flags, .size = sizeof(unsigned int) }
    //     }
    // );
    // EXPECT_EQ(cudaErrorInvalidValue, cuda_retval);

    // // 销毁事件

    // cuda_retval = (cudaError)this->_ws->pos_process( 
    //     /* api_id */ PosApiIndex_cudaEventDestroy, 
    //     /* uuid */ this->_clnt->id,
    //     /* param_desps */ {
    //         { .value = &event, .size = sizeof(cudaEvent_t) }
    //     }
    // );
    // EXPECT_EQ(cudaSuccess, cuda_retval);
} 