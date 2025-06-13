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

#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaMallocFree) {
    uint64_t i;
    cudaError cuda_retval;
    void *mem_ptr = nullptr;
    void **mem_ptr_ = &mem_ptr;
    std::vector<void*> allocated_mem_ptrs;
    std::vector<size_t> mem_sizes({ 
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
        KB(1), KB(2), KB(4), KB(8), KB(16), KB(32), KB(64), KB(128), KB(256), KB(512),
        MB(1), MB(2), MB(4), MB(8), MB(16), MB(32), MB(64), MB(128), MB(256), MB(512)
    });

    for(uint64_t& mem_size : mem_sizes){
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaMalloc, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &mem_ptr_, .size = sizeof(void**) },
                { .value = &mem_size, .size = sizeof(size_t) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
        allocated_mem_ptrs.push_back(mem_ptr);
    }

    for(void*& allocated_mem_ptr : allocated_mem_ptrs){
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaFree, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &allocated_mem_ptr, .size = sizeof(void*) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
    }
}
