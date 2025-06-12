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

#include <cuda_runtime_api.h>

#include "pos/include/common.h"
#include "pos/include/workspace.h"
#include "pos/include/transport.h"
#include "pos/include/parser.h"
#include "pos/cuda_impl/parser_functions.h"

class POSClient_CUDA;

/*!
 *  \brief  POS Parser (CUDA Implementation)
 */
class POSParser_CUDA : public POSParser {
 public:
    POSParser_CUDA(POSWorkspace* ws, POSClient* client) : POSParser(ws, client){}
    ~POSParser_CUDA() = default;
    
 protected:
    /*!
     *  \brief      initialization of the runtime daemon thread
     *  \example    for CUDA, one need to call API e.g. cudaSetDevice first to setup the context for a thread
     */
    pos_retval_t daemon_init() override {
        return POS_SUCCESS; 
    }

    /*!
     *  \brief  insertion of parse functions
     *  \note   this function is implemented in the autogen engine
     *  \return POS_SUCCESS for succefully insertion
     */
    pos_retval_t init_ps_functions() override;
};
