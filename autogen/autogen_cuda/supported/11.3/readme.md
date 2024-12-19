# PhOS Support for CUDA 11.3

This documentation contains `yaml` files for descriptors of CUDA 11.3 APIs for autogenerating processing logic of PhOS parser and worker functions.


## Supported API List

> **Quick Access**
> 1. [CUDA Runtime APIs]()
>       * [Device Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#device-management)
>       * [Error Handling](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#error-handling)
>       * [Event Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#event-management)
>       * [Execution Control](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#execution-control)
>       * [Memory Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#memory-management)
>       * [Occupancy](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#occupancy)
>       * [Stream Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#stream-management)
> 2. [CUDA Driver APIs]()
>       * []()
> 3. [cuBLAS APIs]()
> 4. [cuBLASLt APIs]()
> 5. [cuDNN APIs]()
> 6. [cuSparse APIs]()

### 1. CUDA Runtime APIs

#### [Device Management](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaChooseDevice ( int* device, const cudaDeviceProp* prop )</code><br>
Select compute-device which best matches criteria.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceFlushGPUDirectRDMAWrites ( cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope )</code><br>
Blocks until remote writes are visible to the specified scope.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )</code><br>
Returns information about the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetByPCIBusId ( int* device, const char* pciBusId )</code><br>
Returns a handle to a compute device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaDeviceGetCacheConfig ( cudaFuncCache ** pCacheConfig )</code><br>
Returns the preferred cache configuration for the current device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetDefaultMemPool ( cudaMemPool_t* memPool, int  device )</code><br>
Returns the default mempool of a device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaDeviceGetLimit ( size_t* pValue, cudaLimit limit )</code><br>
Returns resource limits.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetMemPool ( cudaMemPool_t* memPool, int  device )</code><br>
Gets the current mempool for a device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetNvSciSyncAttributes ( void* nvSciSyncAttrList, int  device, int  flags )</code><br>
Return NvSciSync attributes that this device can support.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetP2PAttribute ( int* value, cudaDeviceP2PAttr attr, int  srcDevice, int  dstDevice )</code><br>
Queries attributes of the link between two devices.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetPCIBusId ( char* pciBusId, int  len, int  device )</code><br>
Returns a PCI Bus Id string for the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaDeviceGetSharedMemConfig ( cudaSharedMemConfig ** pConfig )</code><br>
Returns the shared memory configuration for the current device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority )</code><br>
Returns numerical values that correspond to the least and greatest stream priorities.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceGetTexture1DLinearMaxWidth ( size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int  device )</code><br>
Returns the maximum number of elements allocatable in a 1D linear texture for a given element size.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceReset ( void )</code><br>
Destroy all allocations and reset all state on the current device in the current process.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig )</code><br>
Sets the preferred cache configuration for the current device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceSetLimit ( cudaLimit limit, size_t value )</code><br>
Set resource limits.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceSetMemPool ( int  device, cudaMemPool_t memPool )</code><br>
Sets the current memory pool of a device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaDeviceSetSharedMemConfig ( cudaSharedMemConfig config )</code><br>
Sets the shared memory configuration for the current device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaDeviceSynchronize ( void )</code><br>
Wait for compute device to finish.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaGetDevice ( int* device )</code><br>
Returns which device is currently being used.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaGetDeviceCount ( int* count )</code><br>
Returns the number of compute-capable devices.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGetDeviceFlags ( unsigned int* flags )</code><br>
Gets the flags for the current device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )</code><br>
Returns information about the compute-device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaIpcCloseMemHandle ( void* devPtr )</code><br>
Attempts to close memory mapped with cudaIpcOpenMemHandle.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaIpcGetEventHandle ( cudaIpcEventHandle_t* handle, cudaEvent_t event )</code><br>
Gets an interprocess handle for a previously allocated event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr )</code><br>
Gets an interprocess memory handle for an existing device memory allocation.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaIpcOpenEventHandle ( cudaEvent_t* event, cudaIpcEventHandle_t handle )</code><br>
Opens an interprocess event handle for use in the current process.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags )</code><br>
Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaSetDevice ( int  device )</code><br>
Set device to be used for GPU executions.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaSetDeviceFlags ( unsigned int  flags )</code><br>
Sets flags to be used for device executions.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaSetValidDevices ( int* device_arr, int  len )</code><br>
Set a list of devices that can be used for CUDA.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table> 


#### [Error Handling](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>const char* cudaGetErrorName ( cudaError_t error )</code><br>
Returns the string representation of an error code enum name.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>const char* cudaGetErrorString ( cudaError_t error )</code><br>
Returns the description string for an error code.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGetLastError ( void )</code><br>
Returns the last error from a runtime call.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaPeekAtLastError ( void )</code><br>
Returns the last error from a runtime call.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>


#### [Event Management](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventCreate ( cudaEvent_t* event )</code><br>
Creates an event object.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaEventCreateWithFlags ( cudaEvent_t* event, unsigned int  flags )</code><br>
Creates an event object with the specified flags.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaEventDestroy ( cudaEvent_t event )</code><br>
Destroys an event object.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )</code><br>
Computes the elapsed time between events.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventQuery ( cudaEvent_t event )</code><br>
Queries an event's status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )</code><br>
Records an event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventRecordWithFlags ( cudaEvent_t event, cudaStream_t stream = 0, unsigned int  flags = 0 )</code><br>
Records an event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventSynchronize ( cudaEvent_t event )</code><br>
Waits for an event to complete.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>

#### [Execution Control](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t 	cudaFuncGetAttributes ( cudaFuncAttributes* attr, const void* func )</code><br>
Find out attributes for a given function.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaFuncSetAttribute ( const void* func, cudaFuncAttribute attr, int  value )</code><br>
Set attributes for a given function.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaFuncSetCacheConfig ( const void* func, cudaFuncCache cacheConfig )</code><br>
Sets the preferred cache configuration for a device function.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaFuncSetSharedMemConfig ( const void* func, cudaSharedMemConfig config )</code><br>
Sets the shared memory configuration for a device function.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>void* cudaGetParameterBuffer ( size_t alignment, size_t size )</code><br>
Obtains a parameter buffer.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>void* cudaGetParameterBufferV2 ( void* func, dim3 gridDimension, dim3 blockDimension, unsigned int  sharedMemSize )</code><br>
Launches a specified kernel.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaLaunchCooperativeKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )</code><br>
Launches a device function where thread blocks can cooperate and synchronize as they execute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaLaunchCooperativeKernelMultiDevice ( cudaLaunchParams* launchParamsList, unsigned int  numDevices, unsigned int  flags = 0 )</code><br>
Launches device functions on multiple devices where thread blocks can cooperate and synchronize as they execute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaLaunchHostFunc ( cudaStream_t stream, cudaHostFn_t fn, void* userData )</code><br>
Enqueues a host function call in a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )</code><br>
Launches a device function.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaSetDoubleForDevice ( double* d )</code><br>
Converts a double argument to be executed on a device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaSetDoubleForHost ( double* d )</code><br>
Converts a double argument after execution on a device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>


#### [Memory Management](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaArrayGetInfo ( cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array )</code><br>
Gets info about the specified cudaArray.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaArrayGetPlane ( cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int  planeIdx )</code><br>
Gets a CUDA array plane from a CUDA array.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaArrayGetSparseProperties ( cudaArraySparseProperties* sparseProperties, cudaArray_t array )</code><br>
Returns the layout properties of a sparse CUDA array.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaFree ( void* devPtr )</code><br>
Frees memory on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaFreeArray ( cudaArray_t array )</code><br>
Frees an array on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaFreeHost ( void* ptr )</code><br>
Frees page-locked memory.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaFreeMipmappedArray ( cudaMipmappedArray_t mipmappedArray )</code><br>
Frees a mipmapped array on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGetMipmappedArrayLevel ( cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level )</code><br>
Gets a mipmap level of a CUDA mipmapped array.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGetSymbolAddress ( void** devPtr, const void* symbol )</code><br>
Finds the address associated with a CUDA symbol.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGetSymbolSize ( size_t* size, const void* symbol )</code><br>
Finds the size of the object associated with a CUDA symbol.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags )</code><br>
Allocates page-locked memory on the host.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaHostGetDevicePointer ( void** pDevice, void* pHost, unsigned int  flags )</code><br>
Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaHostGetFlags ( unsigned int* pFlags, void* pHost )</code><br>
Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags )</code><br>
Registers an existing host memory range for use by CUDA.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaHostUnregister ( void* ptr )</code><br>
Unregisters a memory range that was registered with cudaHostRegister.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMalloc ( void** devPtr, size_t size )</code><br>
Allocate memory on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMalloc3D ( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent )</code><br>
Allocates logical 1D, 2D, or 3D memory objects on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMalloc3DArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  flags = 0 )</code><br>
Allocate an array on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMallocArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height = 0, unsigned int  flags = 0 )</code><br>
Allocate an array on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMallocHost ( void** ptr, size_t size )</code><br>
Allocates page-locked memory on the host.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal )</code><br>
Allocates memory that will be automatically managed by the Unified Memory system.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMallocMipmappedArray ( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags = 0 )</code><br>
Allocate a mipmapped array on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height )</code><br>
Allocates pitched memory on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemAdvise ( const void* devPtr, size_t count, cudaMemoryAdvise advice, int  device )</code><br>
Advise about the usage of a given memory range.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemGetInfo ( size_t* free, size_t* total )</code><br>
Gets free and total device memory.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemPrefetchAsync ( const void* devPtr, size_t count, int  dstDevice, cudaStream_t stream = 0 )</code><br>
Prefetches memory to the specified destination device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemRangeGetAttribute ( void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count )</code><br>
Query an attribute of a given memory range.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemRangeGetAttributes ( void** data, size_t* dataSizes, cudaMemRangeAttribute ** attributes, size_t numAttributes, const void* devPtr, size_t count )</code><br>
Query attributes of a given memory range.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy2DAsync ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy2DFromArray ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy2DToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy3D ( const cudaMemcpy3DParms* p )</code><br>
Copies data between 3D objects.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p, cudaStream_t stream = 0 )</code><br>
Copies data between 3D objects.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p )</code><br>
Copies memory between devices.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p, cudaStream_t stream = 0 )</code><br>
Copies memory between devices asynchronously.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpyFromSymbol ( void* dst, const void* symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost )</code><br>
Copies data from the given symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpyFromSymbolAsync ( void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data from the given symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpyPeer ( void* dst, int dstDevice, const void* src, int srcDevice, size_t count )</code><br>
Copies memory between two devices.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpyPeerAsync ( void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream = 0 )</code><br>
Copies memory between two devices asynchronously.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice )</code><br>
Copies data to the given symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemcpyToSymbolAsync ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data to the given symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemset ( void* devPtr, int value, size_t count )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemset2D ( void* devPtr, size_t pitch, int value, size_t width, size_t height )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemset2DAsync ( void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0 )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemset3D ( cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemset3DAsync ( cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0 )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMemsetAsync ( void* devPtr, int value, size_t count, cudaStream_t stream = 0 )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaMipmappedArrayGetSparseProperties ( cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap )</code><br>
Returns the layout properties of a sparse CUDA mipmapped array.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaExtent make_cudaExtent ( size_t w, size_t h, size_t d )</code><br>
Returns a cudaExtent based on input parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaPitchedPtr make_cudaPitchedPtr ( void* d, size_t p, size_t xsz, size_t ysz )</code><br>
Returns a cudaPitchedPtr based on input parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaPos make_cudaPos ( size_t x, size_t y, size_t z )</code><br>
Returns a cudaPos based on input parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>

#### [Occupancy](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock ( size_t* dynamicSmemSize, const void* func, int  numBlocks, int  blockSize )</code><br>
Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize )</code><br>
Returns occupancy for a device function.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags )</code><br>
Returns occupancy for a device function with the specified flags.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>


#### [Stream Management](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__STREAM.html)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaCtxResetPersistingL2Cache ( void )</code><br>
Resets all persisting lines in cache to normal status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamAddCallback ( cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags )</code><br>
Add a callback to a compute stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamAttachMemAsync ( cudaStream_t stream, void* devPtr, size_t length = 0, unsigned int  flags = cudaMemAttachSingle )</code><br>
Attach memory to a stream asynchronously.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamBeginCapture ( cudaStream_t stream, cudaStreamCaptureMode mode )</code><br>
Begins graph capture on a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCopyAttributes ( cudaStream_t dst, cudaStream_t src )</code><br>
Copies attributes from source stream to destination stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCreate ( cudaStream_t* pStream )</code><br>
Create an asynchronous stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int  flags )</code><br>
Create an asynchronous stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority )</code><br>
Create an asynchronous stream with the specified priority.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamDestroy ( cudaStream_t stream )</code><br>
Destroys and cleans up an asynchronous stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamEndCapture ( cudaStream_t stream, cudaGraph_t* pGraph )</code><br>
Ends capture on a stream, returning the captured graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetAttribute ( cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out )</code><br>
Queries stream attribute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetCaptureInfo ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus, unsigned long long* pId )</code><br>
Query capture status of a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetCaptureInfo_v2 ( cudaStream_t stream, cudaStreamCaptureStatus ** captureStatus_out, unsigned long long* id_out = 0, cudaGraph_t* graph_out = 0, const cudaGraphNode_t** dependencies_out = 0, size_t* numDependencies_out = 0 )</code><br>
Query a stream's capture state (11.3+).
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetFlags ( cudaStream_t hStream, unsigned int* flags )</code><br>
Query the flags of a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetPriority ( cudaStream_t hStream, int* priority )</code><br>
Query the priority of a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamIsCapturing ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus )</code><br>
Returns a stream's capture status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamQuery ( cudaStream_t stream )</code><br>
Queries an asynchronous stream for completion status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamSetAttribute ( cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value )</code><br>
Sets stream attribute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamSynchronize ( cudaStream_t stream )</code><br>
Waits for stream tasks to complete.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamUpdateCaptureDependencies ( cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int  flags = 0 )</code><br>
Update the set of dependencies in a capturing stream (11.3+).
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamWaitEvent ( cudaStream_t stream, cudaEvent_t event, unsigned int  flags = 0 )</code><br>
Make a compute stream wait on an event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaThreadExchangeStreamCaptureMode ( cudaStreamCaptureMode ** mode )</code><br>
Swaps the stream capture interaction mode for a thread.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>


## Refs:
* [CUDA Toolkit Documentation 11.3](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html)


## TODO:
* we still need to add return value description to support restore recomputation (allocate potential memory space)
