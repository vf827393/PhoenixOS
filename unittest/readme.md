# Unit Test of PhOS

This directory contains UnitTest code for PhOS to verify its correctness of API execution,
program bahavior during checkpoining and after restoring.


## Add UnitTest Case

### 1. API Execution

Take CUDA platform as an instance,
to add a UnitTest for a CUDA API,
one should create a new file with the `unittest/test_cuda/apis/xxx` directory,
where `xxx` is the corresponding library where the API comes from (e.g., `cuda_runtime`, `cuda_driver`, `cublas`).
The file name should be the name of the added API, e.g., `cudaMalloc.cpp`.

An example of UnitTest for `cudaMalloc` is demonstrated as below.
One can directly call `this->_ws->pos_process` to invoke the execution of the targeted API,
and verify the result to be successful after execution.

```cpp
/* unittest/test_cuda/apis/cuda_runtime/cudaMalloc.cpp */

#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaMalloc) {
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
            }ppr
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
        allocated_mem_ptrs.push_back(mem_ptr);
    }
}
```

Note that if the tested API depends on some previous APIs,
e.g., the test of `cudaFree` needs to call `cudaMalloc` first,
one should call `this->_ws->pos_process` to invoke the execution of `cudaMalloc` before `cudaFree`.
The UnitTest for `cudaFree` is demonstrated as below.

```cpp
/* unittest/test_cuda/apis/cuda_runtime/cudaFree.cpp */

#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaFree) {
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

    // NOTE: call cudaMalloc first
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

    // NOTE: call cudaFree then
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
```

Note that if the tested API is an asynchronous API,
one must also call `cudaStreamSynchronize` to verify that the eventual execution of API is successful.
For instance, the UnitTest for `cudaMemcpyAsyncH2D` is demonstrated as below.

```cpp
/* unittest/test_cuda/apis/cuda_runtime/cudaMemcpyAsyncH2D.cpp */

#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaMemcpyAsyncH2D) {
    uint64_t i;
    cudaError cuda_retval;
    void *mem_ptr = nullptr, *host_mem_ptr = nullptr;
    void **mem_ptr_ = &mem_ptr;
    std::vector<void*> allocated_mem_ptrs;
    cudaMemcpyKind kind = cudaMemcpyHostToDevice;
    cudaStream_t stream = 0;
    std::vector<size_t> mem_sizes({ 
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
        KB(1), KB(2), KB(4), KB(8), KB(16), KB(32), KB(64), KB(128), KB(256), KB(512),
        MB(1), MB(2), MB(4), MB(8), MB(16), MB(32), MB(64), MB(128), MB(256), MB(512)
    });

    // malloc GPU memory for copy
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

    // copy data to GPU memory
    for(i=0; i<mem_sizes.size(); i++){
        std::vector<uint8_t> host_memory;
        host_memory.resize(mem_sizes[i]);
        host_mem_ptr = host_memory.data();

        // async copy
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaMemcpyAsyncH2D, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &(allocated_mem_ptrs[i]), .size = sizeof(void*) },
                { .value = &(host_mem_ptr), .size = sizeof(const void*) },
                { .value = &(mem_sizes[i]), .size = sizeof(size_t) },
                { .value = &kind, .size = sizeof(cudaMemcpyKind) },
                { .value = &stream, .size = sizeof(cudaStream_t) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);

        // stream synchronize
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaStreamSynchronize, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &stream, .size = sizeof(cudaStream_t) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
    }
}
```


### 2. Checkpoint

TODO

### 3. Restore

TODO
