# PhOS Describe File for CUDA 11.3

This documentation contains `yaml` files for descriptors of CUDA 11.3 APIs for autogenerating processing logic of PhOS parser and worker functions.

## Supporting API List

> **Quick Access**
> 1. [CUDA Runtime APIs]()
>       * [Device Management]()
>       * [Error Handling]()
>       * [Event Management]()
>       * [Execution Control]()
>       * [Memory Management]()
>       * [Occupancy]()
>       * [Stream Management]()
> 2. [CUDA Driver APIs]()
>       * []()
> 3. [cuBLAS APIs]()
> 4. [cuBLASLt APIs]()
> 5. [cuDNN APIs]()
> 6. [cuSparse APIs]()

### 1. CUDA Runtime APIs

#### [Device Management](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE)

|API|Despcribe|Supported|Test|
|---|---|---|---|
|`cudaError_t cudaChooseDevice ( int* device, const cudaDeviceProp* prop )`|Select compute-device which best matches criteria.| | |
|`cudaError_t cudaDeviceFlushGPUDirectRDMAWrites ( cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope )`|Blocks until remote writes are visible to the specified scope.| | |
|`cudaError_t  cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )`|Returns information about the device.| | |
|`cudaError_t cudaDeviceGetByPCIBusId ( int* device, const char* pciBusId )`|Returns a handle to a compute device.| | |
|`cudaError_t  cudaDeviceGetCacheConfig ( cudaFuncCache ** pCacheConfig )`|Returns the preferred cache configuration for the current device.| | |
|`cudaError_t cudaDeviceGetDefaultMemPool ( cudaMemPool_t* memPool, int  device )`|Returns the default mempool of a device.| | |
|`cudaError_t  cudaDeviceGetLimit ( size_t* pValue, cudaLimit limit )`|Returns resource limits.| | |
|`cudaError_t cudaDeviceGetMemPool ( cudaMemPool_t* memPool, int  device )`|Gets the current mempool for a device.| | |
|`cudaError_t cudaDeviceGetNvSciSyncAttributes ( void* nvSciSyncAttrList, int  device, int  flags )`|Return NvSciSync attributes that this device can support.| | |
|`cudaError_t cudaDeviceGetP2PAttribute ( int* value, cudaDeviceP2PAttr attr, int  srcDevice, int  dstDevice )`|Queries attributes of the link between two devices.| | |
|`cudaError_t cudaDeviceGetPCIBusId ( char* pciBusId, int  len, int  device )`|Returns a PCI Bus Id string for the device.| | |
|`cudaError_t  cudaDeviceGetSharedMemConfig ( cudaSharedMemConfig ** pConfig )`|Returns the shared memory configuration for the current device.| | |
|`cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority )`|Returns numerical values that correspond to the least and greatest stream priorities.| | |
|`cudaError_t cudaDeviceGetTexture1DLinearMaxWidth ( size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int  device )`|Returns the maximum number of elements allocatable in a 1D linear texture for a given element size.| | |
|`cudaError_t cudaDeviceReset ( void )`|Destroy all allocations and reset all state on the current device in the current process.| | |
|`cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig )`|Sets the preferred cache configuration for the current device.| | |
|`cudaError_t cudaDeviceSetLimit ( cudaLimit limit, size_t value )`|Set resource limits.| | |
|`cudaError_t cudaDeviceSetMemPool ( int  device, cudaMemPool_t memPool )`|Sets the current memory pool of a device.| | |
|`cudaError_t cudaDeviceSetSharedMemConfig ( cudaSharedMemConfig config )`|Sets the shared memory configuration for the current device.| | |
|`cudaError_t  cudaDeviceSynchronize ( void )`|Wait for compute device to finish.| | |
|`cudaError_t  cudaGetDevice ( int* device )`|Returns which device is currently being used.| | |
|`cudaError_t  cudaGetDeviceCount ( int* count )`|Returns the number of compute-capable devices.| | |
|`cudaError_t cudaGetDeviceFlags ( unsigned int* flags )`|Gets the flags for the current device.| | |
|`cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )`|Returns information about the compute-device.| | |
|`cudaError_t cudaIpcCloseMemHandle ( void* devPtr )`|Attempts to close memory mapped with cudaIpcOpenMemHandle.| | |
|`cudaError_t cudaIpcGetEventHandle ( cudaIpcEventHandle_t* handle, cudaEvent_t event )`|Gets an interprocess handle for a previously allocated event.| | |
|`cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr )`|Gets an interprocess memory handle for an existing device memory allocation.| | |
|`cudaError_t cudaIpcOpenEventHandle ( cudaEvent_t* event, cudaIpcEventHandle_t handle )`|Opens an interprocess event handle for use in the current process.| | |
|`cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags )`|Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.| | |
|`cudaError_t cudaSetDevice ( int  device )`|Set device to be used for GPU executions.| | |
|`cudaError_t cudaSetDeviceFlags ( unsigned int  flags )`|Sets flags to be used for device executions.| | |
|`cudaError_t cudaSetValidDevices ( int* device_arr, int  len )`|Set a list of devices that can be used for CUDA.| | |



### 2. CUDA Runtime APIs - Error Handling




### 3. CUDA Runtime APIs - Event Management
### 4. CUDA Runtime APIs - Execution Control

### 5. CUDA Runtime APIs - Memory Management
### 6. CUDA Runtime APIs - Occupancy
### 7. CUDA Runtime APIs - Stream Management


## Refs:
* [CUDA Toolkit Documentation 11.3](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html)
* []

## TODO:
* we still need to add return value description to support restore recomputation (allocate potential memory space)
