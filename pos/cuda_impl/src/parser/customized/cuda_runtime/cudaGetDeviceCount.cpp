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

namespace cuda_get_device_count {
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        uint64_t nb_handles;
        int nb_handles_int;

        POSHandleManager_CUDA_Device *hm_device;
        
        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        // obtain handle managers of device
        hm_device = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Device, POSHandleManager_CUDA_Device
        );
        POS_CHECK_POINTER(hm_device);

        nb_handles = hm_device->get_nb_handles();
        nb_handles_int = (int)nb_handles;

        POS_CHECK_POINTER(wqe->api_cxt->ret_data);
        memcpy(
            pos_api_param_addr(wqe, 0),
            &nb_handles_int,
            sizeof(int)
        );

        wqe->status = kPOS_API_Execute_Status_Return_After_Parse;

        return retval;
    }
} // namespace cuda_get_device_count

} // namespace ps_functions
