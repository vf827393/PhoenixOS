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
#include "pos/include/client.h"
#include "pos/cuda_impl/worker.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace wk_functions {

namespace cu_module_load_data {
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle_CUDA_Module *module_handle;
        CUresult res;
        CUmodule module = nullptr, patched_module = nullptr;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        module_handle = reinterpret_cast<POSHandle_CUDA_Module*>(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(module_handle);

        // create normal module
        wqe->api_cxt->return_code = cuModuleLoadData(
            /* module */ &module,
            /* image */  pos_api_param_addr(wqe, 1)
        );
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            module_handle->set_server_addr((void*)module);
        } else {
            POS_WARN("failed to cuModuleLoadData normal module")
        }

        // create patched module
        // wqe->api_cxt->return_code = cuModuleLoadData(
        //     /* module */ &patched_module,
        //     /* image */ (void*)(module_handle->patched_binary.data())
        // );
        // if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
        //     module_handle->patched_server_addr = (void*)(patched_module);
        //     module_handle->mark_status(kPOS_HandleStatus_Active);
        // } else {
        //     POS_WARN("failed to cuModuleLoadData patched module")
        // }

        #if POS_CONF_EVAL_CkptOptLevel > 0 || POS_CONF_EVAL_MigrOptLevel > 0
            // initialize checkpoint bag
            if(unlikely(POS_SUCCESS != (
                retval = module_handle->init_ckpt_bag()
            ))){
                POS_WARN_DETAIL("failed to inilialize checkpoint bag of module handle");
                retval = POS_FAILED;
                goto exit;
            }

            // set host checkpoint record
            if(unlikely(POS_SUCCESS != (
                retval = module_handle->checkpoint_commit_host(
                    /* version_id */ wqe->id,
                    /* data */ pos_api_param_addr(wqe, 1),
                    /* size */ pos_api_param_size(wqe, 1)
                )
            ))){
                POS_WARN_DETAIL("failed to checkpoint commit host of module handle");
                retval = POS_FAILED;
                goto exit;
            }
        #endif

        module_handle->mark_status(kPOS_HandleStatus_Active);

        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace cu_module_load_data

} // namespace wk_functions
