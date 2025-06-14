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

namespace __cuda_register_var {
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *module_handle;
        POSHandle_CUDA_Var *var_handle;
        CUfunction function = NULL;

        CUdeviceptr dptr = 0;
        size_t d_size = 0;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        var_handle = (POSHandle_CUDA_Var*)(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(var_handle);

        POS_ASSERT(var_handle->parent_handles.size() > 0);
        module_handle = var_handle->parent_handles[0];

        wqe->api_cxt->return_code = cuModuleGetGlobal(
            &dptr, &d_size, (CUmodule)(module_handle->server_addr), var_handle->global_name.c_str()
        );

        // record server address
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            var_handle->set_server_addr((void*)dptr);
            var_handle->mark_status(kPOS_HandleStatus_Active);
        }

        // we temp hide the error from this api
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            wqe->api_cxt->return_code = CUDA_SUCCESS;
        }

        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

    exit:
        return retval;
    }
} // namespace __cuda_register_var

} // namespace wk_functions
