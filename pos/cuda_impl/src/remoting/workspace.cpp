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

// TODO(tianle): add metadata for creating new ws here
POSWorkspace_CUDA* pos_create_workspace_cuda(){
    pos_retval_t retval = POS_SUCCESS;
    POSWorkspace_CUDA *pos_cuda_ws = nullptr;

    POS_CHECK_POINTER(pos_cuda_ws = new POSWorkspace_CUDA());
    if(unlikely(POS_SUCCESS != (retval = pos_cuda_ws->init()))){
        POS_WARN("failed to initialize PhOS CUDA Workspace: retval(%u)", retval);
        goto exit;
    }

 exit:
    if(unlikely(retval != POS_SUCCESS)){
        if(pos_cuda_ws != nullptr){ delete pos_cuda_ws; }
        pos_cuda_ws = nullptr;
    }
    return pos_cuda_ws;
}


int pos_destory_workspace_cuda(POSWorkspace_CUDA* pos_cuda_ws){
    int retval = 0;
    pos_retval_t pos_retval = POS_SUCCESS;

    POS_CHECK_POINTER(pos_cuda_ws);

    if(unlikely(POS_SUCCESS != (pos_retval = pos_cuda_ws->deinit()))){
        POS_WARN("failed to deinitialize PhOS CUDA Workspace: retval(%u)", pos_retval);
        retval = 1;
        goto exit;
    }
    delete pos_cuda_ws;

exit:
    return retval;
}


int pos_process(
    POSWorkspace_CUDA *pos_cuda_ws,
    uint64_t api_id,
    uint64_t uuid,
    uint64_t *param_desps,
    int param_num
){
    std::vector<POSAPIParamDesp_t> params(param_num);
    for (int i = 0; i < param_num; i++) {
        POS_CHECK_POINTER((void*)(param_desps[2*i]));
        params[i] = POSAPIParamDesp_t{
            (void*)param_desps[2*i],
            param_desps[2*i+1]
        };
    }
    return pos_cuda_ws->pos_process(api_id, uuid, params);
}


int pos_remoting_stop_query(POSWorkspace_CUDA *pos_cuda_ws, uint64_t uuid){
    int retval = 0;
    volatile POSClient *client;

    POS_CHECK_POINTER(pos_cuda_ws);

    if(unlikely(
        nullptr == (client = pos_cuda_ws->get_client_by_pid(uuid))
    )){
        POS_WARN("try to require access to non-exist client: uuid(%lu)", uuid);
        retval = 0;
        goto exit;
    }
    POS_CHECK_POINTER(client);

    if(unlikely(client->offline_counter > 0)){
        // confirm to the pos worker thread
        if(client->offline_counter == 1){
            POS_DEBUG("confirm client offline: uuid(%lu)", uuid);
            client->offline_counter += 1;
        }
        retval = 1;
        goto exit;
    }

exit:
    return retval;
}


int pos_remoting_stop_confirm(POSWorkspace_CUDA *pos_cuda_ws, uint64_t uuid){
    int retval = 0;
    volatile POSClient *client;

    POS_CHECK_POINTER(pos_cuda_ws);

    if(unlikely(
        nullptr == (client = pos_cuda_ws->get_client_by_pid(uuid))
    )){
        POS_WARN("try to require access to non-exist client: uuid(%lu)", uuid);
        retval = 0;
        goto exit;
    }
    POS_CHECK_POINTER(client);

    POS_ASSERT(client->offline_counter == 1);
    POS_DEBUG("confirm remoting stop: uuid(%lu)", uuid);
    client->offline_counter += 1;

exit:
    return retval;
}


} // extern "C"
