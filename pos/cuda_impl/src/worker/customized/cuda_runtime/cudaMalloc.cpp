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

namespace cuda_malloc {
    // launch function
    POS_WK_FUNC_LAUNCH(){
        CUmemAllocationProp prop = {};
        pos_retval_t retval = POS_SUCCESS;
        POSHandle *memory_handle;
        POSHandle_CUDA_Device *device_handle;
        size_t allocate_size;
        void *ptr;
        CUmemGenericAllocationHandle hdl;
        CUmemAccessDesc access_desc;

        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(wqe);

        device_handle = (POSHandle_CUDA_Device*)(pos_api_input_handle(wqe, 0));
        POS_CHECK_POINTER(device_handle);

        memory_handle = pos_api_create_handle(wqe, 0);
        POS_CHECK_POINTER(memory_handle);

        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_handle->id;

        // create physical memory on the device
        wqe->api_cxt->return_code = cuMemCreate(
            /* handle */ &hdl,
            /* size */ memory_handle->state_size,
            /* prop */ &prop,
            /* flags */ 0
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemCreate: client_addr(%p), state_size(%lu), retval(%d)",
                memory_handle->client_addr, memory_handle->state_size,
                wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // map the virtual memory space to the physical memory
        wqe->api_cxt->return_code = cuMemMap(
            /* ptr */ (CUdeviceptr)(memory_handle->server_addr),
            /* size */ memory_handle->state_size,
            /* offset */ 0ULL,
            /* handle */ hdl,
            /* flags */ 0ULL
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemMap: client_addr(%p), state_size(%lu), retval(%d)",
                memory_handle->client_addr, memory_handle->state_size,
                wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        // set access attribute of this memory
        access_desc.location = prop.location;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        wqe->api_cxt->return_code = cuMemSetAccess(
            /* ptr */ (CUdeviceptr)(memory_handle->server_addr),
            /* size */ memory_handle->state_size,
            /* desc */ &access_desc,
            /* count */ 1ULL
        );
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){
            POS_WARN_DETAIL(
                "failed to execute cuMemSetAccess: client_addr(%p), state_size(%lu), retval(%d)",
                memory_handle->client_addr, memory_handle->state_size,
                wqe->api_cxt->return_code
            );
            retval = POS_FAILED;
            goto exit;
        }

        #if POS_CONF_EVAL_CkptOptLevel > 0 || POS_CONF_EVAL_MigrOptLevel > 0
            // initialize checkpoint bag
            if(unlikely(POS_SUCCESS != (
                retval = memory_handle->init_ckpt_bag()
            ))){
                POS_WARN_DETAIL(
                    "failed to inilialize checkpoint bag of the mamoery handle: client_addr(%p), state_size(%lu)",
                    memory_handle->client_addr, memory_handle->state_size
                );
                retval = POS_FAILED;
                goto exit;
            }
        #endif

        memory_handle->mark_status(kPOS_HandleStatus_Active);

    exit:
        if(unlikely(CUDA_SUCCESS != wqe->api_cxt->return_code)){ 
            POSWorker::__restore(ws, wqe);
        } else {
            POSWorker::__done(ws, wqe);
        }

        return retval;
    }
} // namespace cuda_malloc

} // namespace wk_functions
