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

namespace __cuda_register_function {
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *module_handle;
        POSHandle_CUDA_Function *function_handle;
        CUfunction function = NULL;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);
    
        function_handle = (POSHandle_CUDA_Function*)(pos_api_create_handle(wqe, 0));
        POS_CHECK_POINTER(function_handle);

        POS_ASSERT(function_handle->parent_handles.size() > 0);
        module_handle = function_handle->parent_handles[0];

        wqe->api_cxt->return_code = cuModuleGetFunction(
            &function, (CUmodule)(module_handle->server_addr), function_handle->name.c_str()
        );

        // record server address
        if(likely(CUDA_SUCCESS == wqe->api_cxt->return_code)){
            function_handle->set_server_addr((void*)function);
            function_handle->mark_status(kPOS_HandleStatus_Active);
        }

        // TODO: skip checking
        // if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
        //     POSWorker::__restore(ws, wqe);
        // } else {
        //     POSWorker::__done(ws, wqe);
        // }
        POSWorker::__done(ws, wqe);

    exit:
        return retval;
    }
} // namespace __cuda_register_function

} // namespace wk_functions
