<h1>PhOS Support for CUDA 11.3</h1>

<p>
This documentation contains `yaml` files for descriptors of CUDA 11.3 APIs for autogenerating processing logic of PhOS parser and worker functions.


<h2>1. Supported Data Structures</h2>

<p>
PhOS is supporting various data structures for CUDA 11.3. This section lists all supported data structures.

TODO

<h2>2. Supported API List</h2>

<p>
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
> 7. [NVML]() x3
> 8. [nvRTC]()
> 9. [NCCL]()


<h3>CUDA Runtime APIs</h3>

<ul>
    <li><a href="docs/cudart_device_management.md">Device Management</a></li>
    <li><a href="docs/cudart_error_handling.md">Error Handling</a></li>
    <li><a href="docs/cudart_stream_management.md">Stream Management</a></li>
    <li><a href="docs/cudart_event_management.md">Event Management</a></li>
    <li><a href="docs/cudart_external_resource_interoperability.md">External Resource Interoperability</a></li>
    <li><a href="docs/cudart_execution_control.md">Execution Control</a></li>
    <li><a href="docs/cudart_memory_management.md">Memory Management</a></li>
    <li><a href="docs/cudart_occupancy.md">Occupancy</a></li>
    <li><a href="docs/cudart_stream_ordered_memory_allocator.md">Stream Ordered Memory Allocator</a></li>
    <li><a href="docs/cudart_unified_addressing.md">Unified Addressing</a></li>
    <li><a href="docs/cudart_peer_device_memory_access.md">Peer Device Memory Access</a></li>
</ul>

<p>
Official CUDA Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html

#### [CUDA Driver APIs (0/?)](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/index.html)

TODO



## Refs:
* [CUDA Toolkit Documentation 11.3](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html)


## TODO:
* we still need to add return value description to support restore recomputation (allocate potential memory space)
