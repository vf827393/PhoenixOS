# PhOS Support for CUDA 11.3

This documentation contains `yaml` files for descriptors of CUDA 11.3 APIs for autogenerating processing logic of PhOS parser and worker functions.


## Supported API List

PhOS is supporting popular SDKs on nVIDIA CUDA platforms. This section lists all supporting status of APIS under CUDA 11.3.

> **Quick Access**
> 1. [CUDA Runtime APIs](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#cuda-runtime-apis)
>       * [Device Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#device-management)
>       * [Error Handling](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#error-handling)
>       * [Stream Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#stream-management)
>       * [Event Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#event-management)
>       * [External Resource Interoperability](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#external-resource-interoperability)
>       * [Execution Control](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#execution-control)
>       * [Memory Management](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#memory-management)
>       * [Occupancy](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#occupancy)
>       * [Stream Ordered Memory Allocator](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#stream-ordered-memory-allocator)
>       * [Unified Addressing](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#unified-addressing)
>       * [Peer Device Memory Access](https://github.com/SJTU-IPADS/PhoenixOS/tree/dev/api_support/autogen/autogen_cuda/supported/11.3#peer-device-memory-access)
> 2. [CUDA Driver APIs]()
>       * []()
> 3. [cuBLAS APIs]()
> 4. [cuBLASLt APIs]()
> 5. [cuDNN APIs]()
> 6. [cuSparse APIs]()


### 1. [CUDA Runtime APIs (0/164)](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html)


#### [CUDA Driver APIs (0/?)](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/index.html)

TODO



## Refs:
* [CUDA Toolkit Documentation 11.3](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html)


## TODO:
* we still need to add return value description to support restore recomputation (allocate potential memory space)
