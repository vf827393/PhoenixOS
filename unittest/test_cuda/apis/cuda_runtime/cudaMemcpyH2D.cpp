#include "test_cuda/test_cuda_common.h"

TEST_F(PhOSCudaTest, cudaMemcpyH2D) {
    uint64_t i;
    cudaError cuda_retval;
    void *mem_ptr = nullptr, *host_mem_ptr = nullptr;
    void **mem_ptr_ = &mem_ptr;
    std::vector<void*> allocated_mem_ptrs;
    cudaMemcpyKind kind = cudaMemcpyHostToDevice;
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

        // sync copy
        cuda_retval = (cudaError)this->_ws->pos_process( 
            /* api_id */ PosApiIndex_cudaMemcpyH2D, 
            /* uuid */ this->_clnt->id,
            /* param_desps */ {
                { .value = &(allocated_mem_ptrs[i]), .size = sizeof(void*) },
                { .value = &(host_mem_ptr), .size = sizeof(const void*) },
                { .value = &(mem_sizes[i]), .size = sizeof(size_t) },
                { .value = &kind, .size = sizeof(cudaMemcpyKind) }
            }
        );
        EXPECT_EQ(cudaSuccess, cuda_retval);
    }
}
