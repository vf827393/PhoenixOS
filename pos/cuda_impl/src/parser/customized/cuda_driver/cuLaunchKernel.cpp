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

namespace cu_launch_kernel {
    POS_RT_FUNC_PARSER(){
        pos_retval_t retval = POS_SUCCESS, tmp_retval;
        POSClient_CUDA *client;
        POSHandle_CUDA_Function *function_handle;
        POSHandle_CUDA_Stream *stream_handle;
        POSHandle_CUDA_Memory *memory_handle;

        uint64_t i, j, param_index;
        void *args, *arg_addr, *arg_value;

        uint8_t *struct_base_ptr;
        uint64_t arg_size, struct_offset;

        POSHandleManager_CUDA_Function *hm_function;
        POSHandleManager_CUDA_Stream *hm_stream;
        POSHandleManager_CUDA_Memory *hm_memory;

        /*!
         *  \brief  obtain a potential pointer from a struct by given offset within the struct
         *  \param  base    base address of the struct
         *  \param  offset  offset within the struct
         *  \return potential pointer
         */
        auto __try_get_potential_addr_from_struct_with_offset = [](uint8_t* base, uint64_t offset) -> void* {
            uint8_t *bias_base = base + offset;
            POS_CHECK_POINTER(bias_base);

        #define __ADDR_UNIT(index)   ((uint64_t)(*(bias_base+index) & 0xff) << (index*8))
            return (void*)(
                __ADDR_UNIT(0) | __ADDR_UNIT(1) | __ADDR_UNIT(2) | __ADDR_UNIT(3) | __ADDR_UNIT(4) | __ADDR_UNIT(5)
            );
        #undef __ADDR_UNIT
        };

        /*!
         *  \brief  printing the kernels direction after first parsing
         *  \param  function_handle handler of the function to be printed
         */
        auto __print_kernel_directions = [](POSHandle_CUDA_Function *function_handle){
            POS_CHECK_POINTER(function_handle);
            POS_LOG("obtained direction of kernel %s:", function_handle->signature.c_str());

            // for printing input / output
            auto __unit_print_input_output = [](std::vector<uint32_t>& vec, const char* dir_string){
                uint64_t i, param_index;
                static char param_idx[2048] = {0};
                memset(param_idx, 0, sizeof(param_idx));
                for(i=0; i<vec.size(); i++){
                    param_index = vec[i];
                    if(likely(i!=0)){
                        sprintf(param_idx, "%s, %lu", param_idx, param_index); 
                    } else {
                        sprintf(param_idx, "%lu", param_index);
                    }
                }
                POS_LOG("    %s params: %s", dir_string, param_idx);
            };

            // for printing inout
            auto __unit_print_inout = [](std::vector<std::pair<uint32_t, uint64_t>>& vec, const char* dir_string){
                uint64_t i, struct_offset, param_index;
                static char param_idx[2048] = {0};
                memset(param_idx, 0, sizeof(param_idx));
                for(i=0; i<vec.size(); i++){
                    param_index = vec[i].first;
                    struct_offset = vec[i].second;
                    if(likely(i != 0)){
                        sprintf(param_idx, "%s, %lu(ofs: %lu)", param_idx, param_index, struct_offset); 
                    } else {
                        sprintf(param_idx, "%lu(ofs: %lu)", param_index, struct_offset);
                    }
                };
                POS_LOG("    %s params: %s", dir_string, param_idx);
            };

            __unit_print_input_output(function_handle->input_pointer_params, "input");
            __unit_print_input_output(function_handle->output_pointer_params, "output");
            __unit_print_inout(function_handle->confirmed_suspicious_params, "inout");
        };

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);
        
        // check whether given parameter is valid
        #if POS_CONF_RUNTIME_EnableDebugCheck
            if(unlikely(wqe->api_cxt->params.size() != 6)){
                POS_WARN(
                    "parse(cu_launch_kernel): failed to parse, given %lu params, %lu expected",
                    wqe->api_cxt->params.size(), 6
                );
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
        #endif

        // obtain handle managers of function, stream and memory
        hm_function = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Function, POSHandleManager_CUDA_Function
        );
        POS_CHECK_POINTER(hm_function);

        hm_stream = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Stream, POSHandleManager_CUDA_Stream
        );
        POS_CHECK_POINTER(hm_stream);

        hm_memory = pos_get_client_typed_hm(
            client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // find out the involved function
        retval = hm_function->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 0, uint64_t),
            /* handle */ &function_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_launch_kernel): no function was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 0, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ function_handle
        });

        // find out the involved stream
        retval = hm_stream->get_handle_by_client_addr(
            /* client_addr */ (void*)pos_api_param_value(wqe, 8, uint64_t),
            /* handle */ &stream_handle
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN(
                "parse(cuda_launch_kernel): no stream was founded: client_addr(%p)",
                (void*)pos_api_param_value(wqe, 8, uint64_t)
            );
            goto exit;
        }
        wqe->record_handle<kPOS_Edge_Direction_In>({
            /* handle */ stream_handle
        });

        // the 10th parameter of the API call contains parameter to launch the kernel
        args = pos_api_param_addr(wqe, 9);
        POS_CHECK_POINTER(args);

        // [Cricket Adapt] skip the metadata used by cricket
        // args += (sizeof(size_t) + sizeof(uint16_t) * function_handle->nb_params);
        
        /*!
         *  \note   record all input memory areas
         */
        for(i=0; i<function_handle->input_pointer_params.size(); i++){
            param_index = function_handle->input_pointer_params[i];
 
            arg_addr = args + function_handle->param_offsets[param_index];
            POS_CHECK_POINTER(arg_addr);
            arg_value = *((void**)arg_addr);
            
            /*!
             *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
             *          this is probably normal, so we just ignore this situation
             */
            if(unlikely(arg_value == nullptr)){
                continue;
            }

            tmp_retval = hm_memory->get_handle_by_client_addr(
                /* client_addr */ arg_value,
                /* handle */ &memory_handle
            );

            if(unlikely(tmp_retval != POS_SUCCESS)){
                // POS_WARN(
                //     "%lu(th) parameter of kernel %s is marked as input during kernel parsing phrase, "
                //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                //     param_index, function_handle->signature.c_str(), arg_value
                // );
                continue;
            }

            wqe->record_handle<kPOS_Edge_Direction_In>({
                /* handle */ memory_handle,
                /* param_index */ param_index,
                /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
            });
        }
        
        /*!
         *  \note   record all inout memory areas
         */
        for(i=0; i<function_handle->inout_pointer_params.size(); i++){
            param_index = function_handle->inout_pointer_params[i];

            arg_addr = args + function_handle->param_offsets[param_index];
            POS_CHECK_POINTER(arg_addr);
            arg_value = *((void**)arg_addr);
            
            /*!
             *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
             *          this is probably normal, so we just ignore this situation
             */
            if(unlikely(arg_value == nullptr)){
                continue;
            }

            tmp_retval = hm_memory->get_handle_by_client_addr(
                /* client_addr */ arg_value,
                /* handle */ &memory_handle
            );

            if(unlikely(tmp_retval != POS_SUCCESS)){
                // POS_WARN(
                //     "%lu(th) parameter of kernel %s is marked as inout during kernel parsing phrase, "
                //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                //     param_index, function_handle->signature.c_str(), arg_value
                // );
                continue;
            }

            wqe->record_handle<kPOS_Edge_Direction_InOut>({
                /* handle */ memory_handle,
                /* param_index */ param_index,
                /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
            });

            hm_memory->record_modified_handle(memory_handle);
        }

        /*!
         *  \note   record all output memory areas
         */
        for(i=0; i<function_handle->output_pointer_params.size(); i++){
            param_index = function_handle->output_pointer_params[i];

            arg_addr = args + function_handle->param_offsets[param_index];
            POS_CHECK_POINTER(arg_addr);
            arg_value = *((void**)arg_addr);
            
            /*!
             *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
             *          this is probably normal, so we just ignore this situation
             */
            if(unlikely(arg_value == nullptr)){
                continue;
            }

            tmp_retval = hm_memory->get_handle_by_client_addr(
                /* client_addr */ arg_value,
                /* handle */ &memory_handle
            );

            if(unlikely(tmp_retval != POS_SUCCESS)){
                // POS_WARN(
                //     "%lu(th) parameter of kernel %s is marked as output during kernel parsing phrase, "
                //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                //     param_index, function_handle->signature.c_str(), arg_value
                // );
                continue;
            }

            wqe->record_handle<kPOS_Edge_Direction_Out>({
                /* handle */ memory_handle,
                /* param_index */ param_index,
                /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
            });

            hm_memory->record_modified_handle(memory_handle);
        }

        /*!
         *  \note   check suspicious parameters that might contains pointer
         *  \warn   only check once?
         */
        if(unlikely(function_handle->has_verified_params == false)){          
            for(i=0; i<function_handle->suspicious_params.size(); i++){
                param_index = function_handle->suspicious_params[i];

                // we can skip those already be identified as input / output
                if(std::find(
                    function_handle->input_pointer_params.begin(),
                    function_handle->input_pointer_params.end(),
                    param_index
                ) != function_handle->input_pointer_params.end()){
                    continue;
                }
                if(std::find(
                    function_handle->output_pointer_params.begin(),
                    function_handle->output_pointer_params.end(),
                    param_index
                ) != function_handle->output_pointer_params.end()){
                    continue;
                }

                arg_addr = args + function_handle->param_offsets[param_index];
                POS_CHECK_POINTER(arg_addr);

                struct_base_ptr = (uint8_t*)arg_addr;

                arg_size = function_handle->param_sizes[param_index];
                POS_ASSERT(arg_size >= 6);

                // iterate across the struct using a 8-bytes window
                for(j=0; j<arg_size-6; j++){
                    arg_value = __try_get_potential_addr_from_struct_with_offset(struct_base_ptr, j);

                    tmp_retval = hm_memory->get_handle_by_client_addr(
                        /* client_addr */ arg_value,
                        /* handle */ &memory_handle
                    );
                    if(unlikely(tmp_retval == POS_SUCCESS)){
                        // we treat such memory areas as inout memory
                        function_handle->confirmed_suspicious_params.push_back({
                            /* parameter index */ param_index,
                            /* offset */ j  
                        });

                        wqe->record_handle<kPOS_Edge_Direction_InOut>({
                            /* handle */ memory_handle,
                            /* param_index */ param_index,
                            /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
                        });

                        hm_memory->record_modified_handle(memory_handle);
                    }
                } // foreach arg_size
            } // foreach suspicious_params

            function_handle->has_verified_params = true;
            
            // __print_kernel_directions(function_handle);
        } else {
            for(i=0; i<function_handle->confirmed_suspicious_params.size(); i++){
                param_index = function_handle->confirmed_suspicious_params[i].first;
                struct_offset = function_handle->confirmed_suspicious_params[i].second;

                arg_addr = args + function_handle->param_offsets[param_index];
                POS_CHECK_POINTER(arg_addr);
                arg_value = *((void**)(arg_addr+struct_offset));

                /*!
                 *  \note   sometimes one would launch kernel with some pointer params are nullptr (at least pytorch did),
                 *          this is probably normal, so we just ignore this situation
                 */
                if(unlikely(arg_value == nullptr)){
                    continue;
                }

                tmp_retval = hm_memory->get_handle_by_client_addr(
                    /* client_addr */ arg_value,
                    /* handle */ &memory_handle
                );

                if(unlikely(tmp_retval != POS_SUCCESS)){
                    // POS_WARN(
                    //     "%lu(th) parameter of kernel %s is marked as suspicious output during kernel parsing phrase, "
                    //     "yet it contains non-exist memory address during launching: given client addr(%p)",
                    //     param_index, function_handle->signature.c_str(), arg_value
                    // );
                    continue;
                }

                wqe->record_handle<kPOS_Edge_Direction_InOut>({
                    /* handle */ memory_handle,
                    /* param_index */ param_index,
                    /* offset */ (uint64_t)(arg_value) - (uint64_t)(memory_handle->client_addr)
                });

                hm_memory->record_modified_handle(memory_handle);
            }
        }

    #if POS_CONF_RUNTIME_EnableTrace
        parser->metric_reducers.reduce(
            /* index */ POSParser::KERNEL_in_memories,
            /* value */ function_handle->input_pointer_params.size()
                        + function_handle->inout_pointer_params.size()
        );
        parser->metric_reducers.reduce(
            /* index */ POSParser::KERNEL_out_memories,
            /* value */ function_handle->output_pointer_params.size()
                        + function_handle->inout_pointer_params.size()
        );
        parser->metric_counters.add_counter(
            /* index */ POSParser::KERNEL_number_of_user_kernels
        );
    #endif

    #if POS_PRINT_DEBUG
        // typedef struct __dim3 { uint32_t x; uint32_t y; uint32_t z; } __dim3_t;
        POS_DEBUG(
            "parse(cuda_launch_kernel): function(%s), stream(%p), grid_dim(%u,%u,%u), block_dim(%u,%u,%u), SM_size(%lu)",
            function_handle->name.c_str(), stream_handle->server_addr,
            pos_api_param_addr(wqe, 1),
            pos_api_param_addr(wqe, 2),
            pos_api_param_addr(wqe, 3),
            pos_api_param_addr(wqe, 4),
            pos_api_param_addr(wqe, 5),
            pos_api_param_addr(wqe, 6),
            pos_api_param_value(wqe, 7, size_t)
        );
    #endif

    exit:
        return retval;
    }
} // namespace cu_launch_kernel

} // namespace ps_functions
