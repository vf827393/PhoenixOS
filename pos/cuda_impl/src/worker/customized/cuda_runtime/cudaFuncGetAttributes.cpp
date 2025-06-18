#include <iostream>
#include <cstdint>
#include "pos/include/common.h"
#include "pos/include/client.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "pos/cuda_impl/worker.h"


namespace wk_functions {

namespace cuda_func_get_attributes {    
    POS_WK_FUNC_LAUNCH()
    {
        pos_retval_t retval = POS_SUCCESS;
        struct cudaFuncAttributes *attr = nullptr;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);
        POS_CHECK_POINTER(attr = pos_api_param_value(wqe, 0, struct cudaFuncAttributes *));

        #if POS_CONF_RUNTIME_EnableDebugCheck
            POS_ASSERT(wqe->input_handle_views.size() == 1);
            POS_ASSERT(wqe->output_handle_views.size() == 0);
            POS_ASSERT(wqe->inout_handle_views.size() == 0);
        #endif

        #define GET_FUNC_ATTR(member, name)					                                    \
            do {								                                                \
                int tmp;								                                        \
                wqe->api_cxt->return_code = cuFuncGetAttribute(                                 \
                    &tmp, CU_FUNC_ATTRIBUTE_##name,                                             \
                    (CUfunction)(pos_api_input_handle_offset_server_addr(wqe, 0))               \
                );                                                                              \
                if(unlikely(wqe->api_cxt->return_code != CUDA_SUCCESS)){                        \
                    goto exit;                                                                  \
                }                                                                               \
                attr->member = tmp;						                                        \
            } while(0)
            GET_FUNC_ATTR(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK);
            GET_FUNC_ATTR(sharedSizeBytes, SHARED_SIZE_BYTES);
            GET_FUNC_ATTR(constSizeBytes, CONST_SIZE_BYTES);
            GET_FUNC_ATTR(localSizeBytes, LOCAL_SIZE_BYTES);
            GET_FUNC_ATTR(numRegs, NUM_REGS);
            GET_FUNC_ATTR(ptxVersion, PTX_VERSION);
            GET_FUNC_ATTR(binaryVersion, BINARY_VERSION);
            GET_FUNC_ATTR(cacheModeCA, CACHE_MODE_CA);
            GET_FUNC_ATTR(maxDynamicSharedSizeBytes, MAX_DYNAMIC_SHARED_SIZE_BYTES);
            GET_FUNC_ATTR(preferredShmemCarveout, PREFERRED_SHARED_MEMORY_CARVEOUT);
        #undef GET_FUNC_ATTR

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){
           POSWorker::__restore(ws, wqe);
        } else {
           POSWorker::__done(ws, wqe);
        }

     exit:
        return retval;
    }

} // namespace cuda_func_get_attributes

} // namespace wk_functions
