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

#include "pos/cuda_impl/api_index.h"
#include "pos/cuda_impl/worker_functions.h"
#include "pos/cuda_impl/handle/memory.h"


/*!
 *  \brief  POS Worker (CUDA Implementation)
 */
class POSWorker_CUDA : public POSWorker {
 public:
    POSWorker_CUDA(POSWorkspace* ws, POSClient* client);
    ~POSWorker_CUDA();


    /*!
     *  \brief  make the worker thread synchronized
     *  \param  stream_id   index of the stream to be synced, default to be 0
     */
    pos_retval_t sync(uint64_t stream_id=0) override;


 protected:    
    /*!
     *  \brief      initialization of the worker daemon thread
     *  \example    for CUDA, one need to call API e.g. cudaSetDevice first to setup the context for a thread
     */
    pos_retval_t daemon_init() override;

    
    /*!
     *  \brief  insertion of worker functions
     *  \note   this function is implemented in the autogen engine
     *  \return POS_SUCCESS for succefully insertion
     */
    pos_retval_t init_wk_functions() override;


    /*!
     *  \brief      start an ticker on GPU
     *  \example    on CUDA platform, this API is implemented using ÇUDA event
     *  \param      stream_id   index of the gpu stream to be measured
     *  \return     POS_SUCCESS for successfully started
     */
    pos_retval_t start_gpu_ticker(uint64_t stream_id=0) override;


    /*!
     *  \brief      stop an ticker on GPU
     *  \example    on CUDA platform, this API is implemented using ÇUDA event
     *  \note       this API should cause device synchronization
     *  \param      stream_id   index of the gpu stream to be measured
     *  \param      ticker      value of the ticker
     *  \return     POS_SUCCESS for successfully started
     */
    pos_retval_t stop_gpu_ticker(uint64_t& ticker, uint64_t stream_id=0) override;

 private:
    // ticker events on each CUDA stream
    std::map<cudaStream_t, cudaEvent_t> _cuda_ticker_events;
};
