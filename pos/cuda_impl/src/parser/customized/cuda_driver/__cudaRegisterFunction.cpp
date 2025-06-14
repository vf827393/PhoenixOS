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

namespace __cuda_register_function {
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        uint64_t i;
        POSClient_CUDA *client;
        POSHandle_CUDA_Module *module_handle;
        POSHandle_CUDA_Function *function_handle;
        POSCudaFunctionDesp *function_desp;
        bool found_function_desp;
        POSHandleManager_CUDA_Module *hm_module;
        POSHandleManager_CUDA_Function *hm_function;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 5)){
            POS_WARN(
                "parse(__register_function): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 5
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_module = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Module, POSHandleManager_CUDA_Module
        );
        POS_CHECK_POINTER(hm_module);

        hm_function = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Function, POSHandleManager_CUDA_Function
        );
        POS_CHECK_POINTER(hm_function);

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
                    "parse(__register_function): failed to find module with client address %p",
                    pos_api_param_value(wqe, 0, uint64_t)
                );
                retval = POS_FAILED_NOT_EXIST;
                goto exit;
            }
        }
        
        // check whether the requested kernel name is recorded within the module
        found_function_desp = false;
        for(i=0; i<module_handle->function_desps.size(); i++){
            if(unlikely(
                !strcmp(
                    module_handle->function_desps[i]->name.c_str(),
                    (const char*)(pos_api_param_addr(wqe, 3))
                )
            )){
                function_desp = module_handle->function_desps[i];
                found_function_desp = true;
                break;
            }
        }
        if(unlikely(found_function_desp == false)){
            POS_WARN(
                "parse(__register_function): failed to find function within the module: module_clnt_addr(%p), device_name(%s)",
                pos_api_param_value(wqe, 0, uint64_t), pos_api_param_addr(wqe, 3)
            );
            retval = POS_FAILED_NOT_EXIST;
            goto exit;
        }

        // operate on handler manager
        retval = hm_function->allocate_mocked_resource(
            /* handle */ &function_handle,
            /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({{ 
                /* id */ kPOS_ResourceTypeId_CUDA_Module, 
                /* handles */ std::vector<POSHandle*>({module_handle}) 
            }}),
            /* size */ kPOS_HandleDefaultSize,
            /* use_expected_addr */ true,
            /* expected_addr */ pos_api_param_value(wqe, 1, uint64_t)
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("parse(__register_function): failed to allocate mocked function within the CUDA function handler manager");
            goto exit;
        } else {
            POS_DEBUG(
                "parse(__register_function): allocate mocked function within the CUDA function handler manager: addr(%p), size(%lu), module_server_addr(%p)",
                function_handle->client_addr, function_handle->size,
                module_handle->server_addr
            )
        }

        // transfer function descriptions from the descriptor
        function_handle->nb_params = function_desp->nb_params;
        function_handle->param_offsets = function_desp->param_offsets;
        function_handle->param_sizes = function_desp->param_sizes;
        function_handle->cbank_param_size = function_desp->cbank_param_size;
        function_handle->name = function_desp->name;
        function_handle->input_pointer_params = function_desp->input_pointer_params;
        function_handle->inout_pointer_params = function_desp->inout_pointer_params;
        function_handle->output_pointer_params = function_desp->output_pointer_params;
        function_handle->suspicious_params = function_desp->suspicious_params;
        function_handle->has_verified_params = function_desp->has_verified_params;
        function_handle->confirmed_suspicious_params = function_desp->confirmed_suspicious_params;
        function_handle->signature = function_desp->signature;

        // set handle state as pending to create
        function_handle->mark_status(kPOS_HandleStatus_Create_Pending);

        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_Create>({
            /* handle */ function_handle
        });

        // mark this sync call can be returned after parsing
        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

    exit:
        return retval;
    }
} // namespace __cuda_register_function

} // namespace ps_functions
