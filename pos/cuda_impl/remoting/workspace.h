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

#include "pos/include/command.h"
#include "pos/include/log.h"
#include "pos/include/workspace.h"
#include "pos/cuda_impl/workspace.h"


extern "C" {


/*!
 *  \brief  create new workspace for CUDA platform
 *  \return pointer to the created CUDA workspace
 */
POSWorkspace_CUDA* pos_create_workspace_cuda();


/*!
 *  \brief  destory workspace of CUDA platform
 *  \param  pos_cuda_ws pointer to the CUDA workspace to be destoried
 *  \return 0 for successfully destory
 *          1 for failed
 */
int pos_destory_workspace_cuda(POSWorkspace_CUDA* pos_cuda_ws);


/*!
 *  \brief  execution the callback function of API with specified api_id
 *  \param  pos_cuda_ws pointer to the CUDA workspace
 *  \param  api_id      id of the API to be executed
 *  \param  uuid        uuid of the client
 *  \param  param_desps parameter descriptions, specifiaclly in pairs:
 *                      { pointer to the param, param length }
 *  \param  param_num   number of parameters
 *  \return 0 for successfully destory; else for failed
 */
int pos_process(
    POSWorkspace_CUDA *pos_cuda_ws,
    uint64_t api_id,
    uint64_t uuid,
    uint64_t *param_desps,
    int param_num
);


} // extern "C"
