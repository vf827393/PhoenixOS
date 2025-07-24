#include "test_cuda/test_cuda_common.h"


static void printMatric(std::vector<int> &matrix, int N){
    std::string row_str = "";
    for(int i=0; i<N; i++){
        row_str = "";
        for(int j=0; j<N; j++){
            row_str += std::format("{} ", matrix[N*i+j]);
        }
        printf("%s\n", row_str.c_str());
    }
}

static bool verifyMatrixMultiplicationResult(
    std::vector<int> &matrix_A, 
    std::vector<int> &matrix_B, 
    std::vector<int> &matrix_C,
    int N
  ){
    std::vector<int> matrix_correct;

    // precaluclate correct matrix
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            int tmp=0;
            for(int k=0; k<N; k++){
                tmp += matrix_A[N*i+k]*matrix_B[N*k+j];
            }
            matrix_correct.push_back(tmp);
        }
    }

    // for every row of matrix_C
    for(int i=0; i<N; i++){
        // for every column of matrix_C
        for(int j=0; j<N; j++){
            // assertion
            if(matrix_correct[i*N+j] != matrix_C[i*N+j]){
                POS_WARN(
                    "matmul not matched: "
                    "row_id(%d), col_id(%d), "
                    "expected(%d), obtain(%d)",
                    i, j,
                    matrix_correct[i*N+j], matrix_C[i*N+j]
                );
                POS_WARN("A:");
                    printMatric(matrix_A, N);
                POS_WARN("B:");
                    printMatric(matrix_B, N);
                POS_WARN("C:");
                    printMatric(matrix_C, N);
                POS_WARN("correct C:");
                    printMatric(matrix_correct, N);
                return false;
            }
        }
    }

    return true;
}

TEST_F(PhOSCudaTest, cuLaunchKernel) {
    cudaError cuda_rt_retval;
    CUresult cuda_dv_retval;
    CUmodule module;
    CUmodule *module_ptr = &module;
    CUfunction function;
    CUfunction *function_ptr = &function;
    CUstream stream;
    std::ifstream in;
    std::stringstream buffer;
    std::string function_name;
    const char* function_name_ptr;
    cudaMemcpyKind kind = cudaMemcpyHostToDevice;

    std::vector<int> matrix_A, matrix_B, matrix_C;
    int *hmem_A = nullptr, *hmem_B = nullptr, *hmem_C = nullptr;
    int *dmem_A = nullptr, *dmem_B = nullptr, *dmem_C = nullptr;
    int **dmem_A_ptr = &dmem_A, **dmem_B_ptr = &dmem_B, **dmem_C_ptr = &dmem_C;
    int N = 8;
    uint64_t mem_size = N * N * sizeof(int);

    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
    void *list_params = nullptr, *list_extra = nullptr;
    uint64_t list_params_size = 0;

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path current_abs_path = std::filesystem::absolute(current_path);
    std::filesystem::path current_dir_abs_path = current_abs_path.parent_path();
    std::filesystem::path current_dir_dir_abs_path = current_dir_abs_path.parent_path();

    #if CUDA_VERSION >= 9000 && CUDA_VERSION < 11040
        std::filesystem::path cubin_asb_path = std::filesystem::canonical(
            current_dir_dir_abs_path / "assets" / "sm70_72_75_80_86.fatbin"
        );
    #else
        POS_WARN("no test file for current cuda architecture: cuda_version(%d)", CUDA_VERSION);
        goto exit;
    #endif

    in.open(cubin_asb_path, std::ios::binary);
    EXPECT_EQ(true, in.is_open());
    buffer << in.rdbuf();

    // load module first
    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuModuleLoadData, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &module_ptr, .size = sizeof(CUmodule*) },
            { .value = buffer.str().data(), .size = buffer.str().size() }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    function_name = "_Z15squareMatrixMulPKiS0_Pii";
    function_name_ptr = function_name.data();
    
    // get function
    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuModuleGetFunction, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &function_ptr, .size = sizeof(CUfunction*) },
            { .value = &module, .size = sizeof(CUmodule) },
            { .value = &function_name_ptr, .size = sizeof(const char*) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    // allocate memory for computation
    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaMalloc, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &dmem_A_ptr, .size = sizeof(void**) },
            { .value = &mem_size, .size = sizeof(uint64_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaMalloc, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &dmem_B_ptr, .size = sizeof(void**) },
            { .value = &mem_size, .size = sizeof(uint64_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaMalloc, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &dmem_C_ptr, .size = sizeof(void**) },
            { .value = &mem_size, .size = sizeof(uint64_t) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    // initialize values in test matrices
    matrix_A.resize(N*N);
    matrix_B.resize(N*N);
    matrix_C.resize(N*N);
    for (int i=0; i<N*N; i++){
        matrix_A[i] = rand()%100;
        matrix_B[i] = rand()%100;
    }
    hmem_A = matrix_A.data();
    hmem_B = matrix_B.data();
    hmem_C = matrix_C.data();

    // copy matrix A
    kind = cudaMemcpyHostToDevice;
    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaMemcpyH2D, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &(dmem_A), .size = sizeof(void*) },
            { .value = &(hmem_A), .size = sizeof(const void*) },
            { .value = &(mem_size), .size = sizeof(size_t) },
            { .value = &kind, .size = sizeof(cudaMemcpyKind) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    // copy matrix B
    kind = cudaMemcpyHostToDevice;
    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaMemcpyH2D, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &(dmem_B), .size = sizeof(void*) },
            { .value = &(hmem_B), .size = sizeof(const void*) },
            { .value = &(mem_size), .size = sizeof(size_t) },
            { .value = &kind, .size = sizeof(cudaMemcpyKind) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    // formup parameters list of the kernel
    list_params_size = sizeof(dmem_A) + sizeof(dmem_B) + sizeof(dmem_C) + sizeof(N);
    POS_CHECK_POINTER(list_params = malloc(list_params_size));
    memcpy(list_params, &dmem_A, sizeof(dmem_A));
    memcpy(list_params + sizeof(dmem_A), &dmem_B, sizeof(dmem_B));
    memcpy(list_params + sizeof(dmem_A) + sizeof(dmem_B), &dmem_C, sizeof(dmem_C));
    memcpy(list_params + sizeof(dmem_A) + sizeof(dmem_B) + sizeof(dmem_C), &N, sizeof(N));

    // formup launching parameters
    stream = 0;
    gridDimX = 1;
    gridDimY = 1;
    gridDimZ = 1;
    blockDimX = 32;
    blockDimY = 32;
    blockDimZ = 1;
    sharedMemBytes = 0;

    // launch kernel
    cuda_dv_retval = (CUresult)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuLaunchKernel, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &function, .size = sizeof(CUfunction) },
            { .value = &gridDimX, .size = sizeof(unsigned int) },
            { .value = &gridDimY, .size = sizeof(unsigned int) },
            { .value = &gridDimZ, .size = sizeof(unsigned int) },
            { .value = &blockDimX, .size = sizeof(unsigned int) },
            { .value = &blockDimY, .size = sizeof(unsigned int) },
            { .value = &blockDimZ, .size = sizeof(unsigned int) },
            { .value = &sharedMemBytes, .size = sizeof(unsigned int) },
            { .value = &stream, .size = sizeof(CUstream) },
            { .value = list_params, .size = list_params_size },
            { .value = list_extra, .size = 0 }
        }
    );
    EXPECT_EQ(CUDA_SUCCESS, cuda_dv_retval);

    // we can free list_params right after launch kernel
    free(list_params);

    // stream synchronize
    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaStreamSynchronize, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &stream, .size = sizeof(CUstream) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    // copy matrix C
    kind = cudaMemcpyDeviceToHost;
    cuda_rt_retval = (cudaError)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cudaMemcpyD2H, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ {
            { .value = &(hmem_C), .size = sizeof(void*) },
            { .value = &(dmem_C), .size = sizeof(const void*) },
            { .value = &(mem_size), .size = sizeof(size_t) },
            { .value = &kind, .size = sizeof(cudaMemcpyKind) }
        }
    );
    EXPECT_EQ(cudaSuccess, cuda_rt_retval);

    // verify matmul result
    EXPECT_EQ(verifyMatrixMultiplicationResult(matrix_A, matrix_B, matrix_C, N), true);

exit:
    if(in.is_open()){
        in.close();
    }
}
