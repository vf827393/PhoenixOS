/*
 * Copyright 2025 The PhoenixOS Authors. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <thread>
#include <future>

#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/transport.h"
#include "pos/include/worker.h"
#include "pos/include/checkpoint.h"
#include "pos/cuda_impl/handle/memory.h"
#include "pos/cuda_impl/worker.h"


POSWorker_CUDA::POSWorker_CUDA(POSWorkspace* ws, POSClient* client)
    : POSWorker(ws, client) {}


POSWorker_CUDA::~POSWorker_CUDA(){}


pos_retval_t POSWorker_CUDA::sync(uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cuda_rt_retval;

    cuda_rt_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
    if(unlikely(cuda_rt_retval != cudaSuccess)){
        POS_WARN_C_DETAIL(
            "failed to synchronize worker, is this a bug?: stream_id(%p), cuda_rt_retval(%d)",
            stream_id, cuda_rt_retval
        );
        retval = POS_FAILED;
    }

    return retval;
}


pos_retval_t POSWorker_CUDA::daemon_init(){
    /*!
        *  \note   make sure the worker thread is bound to a CUDA context
        *          if we don't do this and use the driver API, it might be unintialized
        */
    if(cudaSetDevice(0) != cudaSuccess){
        POS_WARN_C_DETAIL("worker thread failed to invoke cudaSetDevice");
        return POS_FAILED; 
    }
    cudaDeviceSynchronize();
    
#if POS_CONF_EVAL_CkptOptLevel == 2
    POS_ASSERT(
        cudaSuccess == cudaStreamCreate((cudaStream_t*)(&this->_ckpt_stream_id))
    );

    POS_ASSERT(
        cudaSuccess == cudaStreamCreate((cudaStream_t*)(&this->_cow_stream_id))
    );
#endif

#if POS_CONF_EVAL_MigrOptLevel == 2
    POS_ASSERT(
        cudaSuccess == cudaStreamCreate((cudaStream_t*)(&this->_migration_precopy_stream_id))
    );
#endif

    return POS_SUCCESS; 
}


pos_retval_t POSWorker_CUDA::start_gpu_ticker(uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    cudaError_t cudart_retval;
    cudaEvent_t start = (cudaEvent_t)(nullptr);

    if(unlikely(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) > 0)){
        POS_WARN_C("start duplicated gpu ticker on the same CUDA stream, overwrite");
        cudart_retval = cudaEventDestroy(this->_cuda_ticker_events[(cudaStream_t)(stream_id)]);
        if(unlikely(cudart_retval != CUDA_SUCCESS)){
            POS_WARN_C("failed to destory old ticker CUDA event");
        }
    }

    cudart_retval = cudaEventCreate(&start);
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to create new ticker CUDA event");
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaEventRecord(start, (cudaStream_t)(stream_id));
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to start event record on specified stream: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    this->_cuda_ticker_events[(cudaStream_t)(stream_id)] = start;

exit:
    if(retval != POS_SUCCESS){
        if(start != (cudaEvent_t)(nullptr)){
            cudaEventDestroy(start);
        }
            
        if(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) > 0){
            this->_cuda_ticker_events.erase((cudaStream_t)(stream_id));
        }
    }
    return retval;
}


pos_retval_t POSWorker_CUDA::stop_gpu_ticker(uint64_t& ticker, uint64_t stream_id){
    pos_retval_t retval = POS_SUCCESS;
    float duration_ms = 0;
    cudaError_t cudart_retval;
    cudaEvent_t stop = (cudaEvent_t)(nullptr);

    if(unlikely(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) == 0)){
        POS_WARN_C("failed to stop gpu ticker, no start event exists");
        retval = POS_FAILED_NOT_EXIST;
        goto exit;
    }

    cudart_retval = cudaEventCreate(&stop);
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to create new ticker CUDA event");
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaEventRecord(stop, (cudaStream_t)(stream_id));
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to start event record on specified stream: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaStreamSynchronize((cudaStream_t)(stream_id));
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to sync specified stream: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    cudart_retval = cudaEventElapsedTime(
        &duration_ms, this->_cuda_ticker_events[(cudaStream_t)(stream_id)], stop
    );
    if(unlikely(cudart_retval != CUDA_SUCCESS)){
        POS_WARN_C("failed to elapsed time between CUDA events: stream_id(%lu)", stream_id);
        retval = POS_FAILED;
        goto exit;
    }

    POS_CHECK_POINTER(this->_ws);
    ticker = (uint64_t)(this->_ws->tsc_timer.ms_to_tick((uint64_t)(duration_ms)));

exit:
    if(this->_cuda_ticker_events.count((cudaStream_t)(stream_id)) > 0){
        if(this->_cuda_ticker_events[(cudaStream_t)(stream_id)] != (cudaEvent_t)(nullptr)){
            cudaEventDestroy(this->_cuda_ticker_events[(cudaStream_t)(stream_id)]);
        }
        
        this->_cuda_ticker_events.erase((cudaStream_t)(stream_id));
    }
    if(stop != (cudaEvent_t)(nullptr)){
        cudaEventDestroy(stop);
    }

    return retval;
}
