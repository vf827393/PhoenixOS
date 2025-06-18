#include <iostream>
#include "pos/include/common.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "pos/cuda_impl/handle.h"
#include "pos/cuda_impl/parser.h"
#include "pos/cuda_impl/client.h"
#include "pos/cuda_impl/api_context.h"


namespace ps_functions
{
namespace cuda_malloc
{
    POS_RT_FUNC_PARSER()
    {
        pos_retval_t retval = POS_SUCCESS;
        POSClient_CUDA *client;
        POSHandleManager_CUDA_Context *hm_context;
        POSHandle_CUDA_Context *context_handle_0;
        POSHandleManager_CUDA_Memory *hm_memory;
        POSHandle_CUDA_Memory *memory_handle_0;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        client = (POSClient_CUDA*)(wqe->client);
        POS_CHECK_POINTER(client);

        #if POS_CONF_RUNTIME_EnableDebugCheck
            // check whether given parameter is valid
           if(unlikely(wqe->api_cxt->params.size() != 2)) {
               POS_WARN(
                   "parse(cuda_malloc): failed to parse, given %lu params, 2 expected",
                   wqe->api_cxt->params.size()
               );
               retval = POS_FAILED_INVALID_INPUT;
               goto exit;
           }
        #endif

        // obtain handle manager of kPOS_ResourceTypeId_CUDA_Context
        hm_context = pos_get_client_typed_hm(
           client, kPOS_ResourceTypeId_CUDA_Context, POSHandleManager_CUDA_Context
        );
        POS_CHECK_POINTER(hm_context);

        // obtain handle from hm (use latest used handle)
        POS_CHECK_POINTER(context_handle_0 = hm_context->latest_used_handle);

        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_In>({
           /* handle */ context_handle_0
        });

        // obtain handle manager of kPOS_ResourceTypeId_CUDA_Memory
        hm_memory = pos_get_client_typed_hm(
           client, kPOS_ResourceTypeId_CUDA_Memory, POSHandleManager_CUDA_Memory
        );
        POS_CHECK_POINTER(hm_memory);

        // create handle in the hm
        retval = hm_memory->allocate_mocked_resource(
           /* handle */ &memory_handle_0,
           /* related_handles */ std::map<uint64_t, std::vector<POSHandle*>>({
                {
                    /* id */ kPOS_ResourceTypeId_CUDA_Context,
                    /* handles */ std::vector<POSHandle*>({
                         context_handle_0
                    })
                }
           }),
           /* size */ pos_api_param_value(wqe, 1, uint64_t),
           /* use_expected_addr */ false,
           /* expected_addr */ 0,
           /* state_size */ pos_api_param_value(wqe, 1, uint64_t)
        );
        if(unlikely(retval != POS_SUCCESS)){
           POS_WARN("parse(cuda_malloc): failed to allocate mocked POSHandle_CUDA_Memory resource within the handler manager");
           memset(pos_api_param_value(wqe, 0, void**), 0, sizeof(uint64_t));
           goto exit;
        } else {
           memcpy(pos_api_param_value(wqe, 0, void**), &(memory_handle_0->client_addr), sizeof(uint64_t));
        }

        // record the related handle to QE
        wqe->record_handle<kPOS_Edge_Direction_Create>({
           /* handle */ memory_handle_0,
           /* param_index */ 0,
           /* offset */ pos_api_param_value(wqe, 0, uint64_t) - (uint64_t)(memory_handle_0->client_addr)
        });

     exit:

        return retval;
    }

} // namespace cuda_malloc

} // namespace ps_functions
