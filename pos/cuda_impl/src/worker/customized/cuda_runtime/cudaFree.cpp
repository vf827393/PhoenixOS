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

namespace cuda_free {
    POS_WK_FUNC_LAUNCH(){
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle;
        CUmemGenericAllocationHandle hdl;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        memory_handle = pos_api_delete_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        // obtain the physical memory handle
        wqe->api_cxt->return_code = cuMemRetainAllocationHandle(&hdl, memory_handle->server_addr);
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemRetainAllocationHandle: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // ummap the virtual memory
        wqe->api_cxt->return_code = cuMemUnmap(
            /* ptr */ (CUdeviceptr)(memory_handle->server_addr),
            /* size */ memory_handle->state_size
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemUnmap: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // release the physical memory
        wqe->api_cxt->return_code = cuMemRelease(hdl);
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemRelease x 1: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // as we call cuMemRetainAllocationHandle above, we need to release again
        wqe->api_cxt->return_code = cuMemRelease(hdl);
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemRelease x 2: client_addr(%p), retval(%d)",
                memory_handle->client_addr, wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        memory_handle->mark_status(kPOS_HandleStatus_Deleted);
        
    exit:
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_free

} // namespace wk_functions
