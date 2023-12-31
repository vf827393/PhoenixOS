#include <iostream>
#include <vector>

#include <stdint.h>

#include "cpu_rpc_prot.h"

#include "pos/common.h"
#include "pos/api_context.h"

/*!
 *  \brief  all POS-hijacked should be record here, used for debug checking
 */
const std::vector<uint64_t> pos_hijacked_apis({
    /* CUDA Runtime */
    CUDA_MALLOC,
    CUDA_LAUNCH_KERNEL,
    CUDA_MEMCPY_HTOD,
    CUDA_MEMCPY_DTOH,
    CUDA_MEMCPY_DTOD,
    CUDA_MEMCPY_HTOD_ASYNC,
    CUDA_MEMCPY_DTOH_ASYNC,
    CUDA_MEMCPY_DTOD_ASYNC,
    CUDA_SET_DEVICE,
    CUDA_GET_LAST_ERROR,
    CUDA_GET_ERROR_STRING,
    CUDA_GET_DEVICE_COUNT,
    CUDA_GET_DEVICE_PROPERTIES,
    CUDA_GET_DEVICE,
    CUDA_STREAM_SYNCHRONIZE,
    CUDA_STREAM_IS_CAPTURING,
    CUDA_EVENT_CREATE_WITH_FLAGS,
    CUDA_EVENT_DESTROY,
    CUDA_EVENT_RECORD,

    /* CUDA Driver */
    rpc_cuModuleLoad,
    rpc_cuModuleGetFunction,
    rpc_register_var,
    rpc_cuDevicePrimaryCtxGetState,
    
    /* cuBLAS */
    rpc_cublasCreate,
    rpc_cublasSetStream,
    rpc_cublasSetMathMode,
    rpc_cublasSgemm,

    /* no need to hijack */
    rpc_printmessage,
    rpc_elf_load,
    rpc_register_function,
    rpc_elf_unload,
    rpc_deinit
});
