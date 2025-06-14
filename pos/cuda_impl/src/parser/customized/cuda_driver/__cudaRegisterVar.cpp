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

#include <iostream>

#include "pos/include/common.h"
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/parser.h"
#include "pos/cuda_impl/client.h"
#include "pos/cuda_impl/api_context.h"


namespace ps_functions {

namespace __cuda_register_var {
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Module *module_handle;
        POSHandle_CUDA_Var *var_handle;
        POSHandleManager_CUDA_Module *hm_module;
        POSHandleManager_CUDA_Var *hm_var;
        POSClient_CUDA *client;
        
        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
        #if POS_CONF_RUNTIME_EnableDebugCheck
            if(unlikely(wqe->api_cxt->params.size() != 8)){
                POS_WARN(
                    "parse(cu_module_get_function): failed to parse, given %lu params, %lu expected",
                    wqe->api_cxt->params.size(), 8
                );
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
        #endif

        hm_module = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Module, POSHandleManager_CUDA_Module
        );
        POS_CHECK_POINTER(hm_module);

        hm_var = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Var, POSHandleManager_CUDA_Var
        );
        POS_CHECK_POINTER(hm_var);

        // obtain target handle
        if(likely(
            hm_module->latest_used_handle->is_client_addr_in_range((void*)pos_api_param_value(wqe, 0, uint64_t))
        )){
            module_handle = hm_module->latest_used_handle;
        } else {
            if(unlikely(
                POS_SUCCESS != hm_module->get_handle_by_client_addr(
                    /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
                    /* handle */&module_handle
                )
            )){
                POS_WARN(
                    "parse(cu_module_get_global): failed to find module with client address %p",
                    pos_api_param_value(wqe, 0, uint64_t)
                );
                retval = POS_FAILED_NOT_EXIST;
                goto exit;
            }
        }

        // allocate a new var within the manager
        retval = hm_var->allocate_mocked_resource(
            /* handle */ &var_handle,
            /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({{ 
                /* id */ kPOS_ResourceTypeId_CUDA_Module, 
                /* handles */ std::vector<POSHandle*>({module_handle}) 
            }}),
            /* size */ sizeof(CUdeviceptr),
            /* use_expected_addr */ true,
            /* expected_addr */ pos_api_param_value(wqe, 1, uint64_t)
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("parse(cu_module_get_global): failed to allocate mocked var within the CUDA var handler manager");
            goto exit;
        } else {
            POS_DEBUG(
                "parse(cu_module_get_global): allocate mocked var within the CUDA var handler manager: addr(%p), size(%lu), module_server_addr(%p)",
                var_handle->client_addr, var_handle->size, module_handle->server_addr
            )
        }

        // record the name of the var in the handle
        var_handle->global_name = std::string((const char*)(pos_api_param_addr(wqe, 3)));

        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_Create>({
            /* handle */ var_handle
        });

        // mark this sync call can be returned after parsing
        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

    exit:
        return retval;
    }
} // namespace __cuda_register_var

} // namespace ps_functions
