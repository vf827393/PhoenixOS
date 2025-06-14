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

namespace cu_module_load_data {
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        uint64_t i;
        POSClient_CUDA *client;
        POSHandle_CUDA_Module *module_handle;
        POSHandle_CUDA_Function *function_handle;
        POSHandleManager_CUDA_Context *hm_context;
        POSHandleManager_CUDA_Module *hm_module;
        POSHandleManager_CUDA_Function *hm_function;

    #define __POS_DUMP_FATBIN 0
    #if __POS_DUMP_FATBIN 
        std::ofstream fatbin_file("/tmp/fatbin.bin", std::ios::binary);
        if(unlikely(!fatbin_file)){
            POS_ERROR_DETAIL("failed to open /tmp/fatbin.bin");
        }

        std::ofstream fatbin_patch_file("/tmp/fatbin_patch.bin", std::ios::binary);
        if(unlikely(!fatbin_patch_file)){
            POS_ERROR_DETAIL("failed to open /tmp/fatbin_patch.bin");
        }

        fatbin_file.write((const char*)(pos_api_param_addr(wqe, 0)), pos_api_param_size(wqe, 0));
        fatbin_file.flush();
        fatbin_file.close();
    #endif

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);
    
        // check whether given parameter is valid
    #if POS_CONF_RUNTIME_EnableDebugCheck
        if(unlikely(wqe->api_cxt->params.size() != 1)){
            POS_WARN(
                "parse(cu_module_load_data): failed to parse, given %lu params, %lu expected",
                wqe->api_cxt->params.size(), 1
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }
    #endif

        hm_context = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Context, POSHandleManager_CUDA_Context
        );
        POS_CHECK_POINTER(hm_context);
        POS_CHECK_POINTER(hm_context->latest_used_handle);

        hm_module = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Module, POSHandleManager_CUDA_Module
        );
        POS_CHECK_POINTER(hm_module);

        hm_function = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Function, POSHandleManager_CUDA_Function
        );
        POS_CHECK_POINTER(hm_function);

        // operate on handler manager
        retval = hm_module->allocate_mocked_resource(
            /* handle */ &module_handle,
            /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({{ 
                /* id */ kPOS_ResourceTypeId_CUDA_Context, 
                /* handles */ std::vector<POSHandle*>({hm_context->latest_used_handle}) 
            }}),
            /* size */ kPOS_HandleDefaultSize,
            /* use_expected_addr */ false,
            /* expected_addr */ 0,
            /* state_size */ pos_api_param_size(wqe, 0)
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("parse(cu_module_load_data): failed to allocate mocked module within the CUDA module handler manager");
            goto exit;
        } else {
            POS_DEBUG(
                "parse(cu_module_load_data): allocate mocked module within the CUDA module handler manager: addr(%p), size(%lu), context_server_addr(%p)",
                module_handle->client_addr, module_handle->size,
                hm_context->latest_used_handle->server_addr
            );
            POS_CHECK_POINTER(wqe->api_cxt->ret_data);
            memcpy(wqe->api_cxt->ret_data, &(module_handle->client_addr), sizeof(void*));
        }

        // set current handle as the latest used handle
        hm_module->latest_used_handle = module_handle;

        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_Create>({
            /* handle */ module_handle
        });

        // analyse the fatbin and stores the function attributes in the handle
        retval = POSUtil_CUDA_Fatbin::obtain_functions_from_cuda_binary(
            /* binary_ptr */ (uint8_t*)(pos_api_param_addr(wqe, 0)),
            /* binary_size */ pos_api_param_size(wqe, 0),
            /* deps */ &(module_handle->function_desps),
            /* cached_desp_map */ hm_module->cached_function_desps
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cu_module_load_data): failed to parse fatbin/cubin"
            );
        }

        // patched PTX within the fatbin
        // retval = POSUtil_CUDA_Kernel_Patcher::patch_fatbin_binary(
        //     /* binary_ptr */ (uint8_t*)(pos_api_param_addr(wqe, 0)),
        //     /* patched_binary */ module_handle->patched_binary
        // );
        // if(unlikely(retval != POS_SUCCESS)){
        //     POS_WARN(
        //         "parse(cu_module_load_data): failed to patch PTX within the fatbin with %lu functions",
        //         module_handle->function_desps.size()
        //     );
        // }
        // POS_LOG(
        //     "parse(cu_module_load_data): patched %lu functions in the fatbin",
        //     module_handle->function_desps.size()
        // );

    #if __POS_DUMP_FATBIN
        fatbin_patch_file.write((const char*)(module_handle->patched_binary.data()), module_handle->patched_binary.size());
        fatbin_patch_file.flush();
        fatbin_patch_file.close();
    #endif

        #if POS_PRINT_DEBUG
            for(auto desp : module_handle->function_desps){
                char *offsets_info = (char*)malloc(1024); POS_CHECK_POINTER(offsets_info);
                char *sizes_info = (char*)malloc(1024); POS_CHECK_POINTER(sizes_info);
                memset(offsets_info, 0, 1024);
                memset(sizes_info, 0, 1024);

                for(auto offset : desp->param_offsets){ sprintf(offsets_info, "%s, %u", offsets_info, offset); }
                for(auto size : desp->param_sizes){ sprintf(sizes_info, "%s, %u", sizes_info, size); }

                POS_DEBUG(
                    "function_name(%s), offsets(%s), param_sizes(%s)",
                    desp->name.c_str(), offsets_info, sizes_info
                );

                free(offsets_info);
                free(sizes_info);
            }
        #endif
        
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN("parse(cu_module_load_data): failed to launch op");
            goto exit;
        }

        // mark this sync call can be returned after parsing
        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

    #undef __POS_DUMP_FATBIN

    exit:
        return retval;
    }
} // namespace cu_module_load_data

} // namespace ps_functions
