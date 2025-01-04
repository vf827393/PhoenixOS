<h1>PhOS Support for CUDA 11.3</h1>

<p>
This directory contains `yaml` files which contain descriptors of CUDA 11.3 APIs for autogenerating processing logic of PhOS parser and worker functions.

<p>
PhOS is supporting popular SDKs on nVIDIA CUDA platforms. Below we list supporting status of all data structures and APIs under CUDA 11.3.


<h3>CUDA Runtime APIs</h3>
<p>
Reference: Official CUDA Documentation for Runtime APIs: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html
<p>
Below lists supported status of each API under different categories.
<ul>
    <li><a href="docs/cuda_runtime/cudart_device_management.md">Device Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_error_handling.md">Error Handling</a></li>
    <li><a href="docs/cuda_runtime/cudart_stream_management.md">Stream Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_event_management.md">Event Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_external_resource_interoperability.md">External Resource Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_execution_control.md">Execution Control</a></li>
    <li><a href="docs/cuda_runtime/cudart_memory_management.md">Memory Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_occupancy.md">Occupancy</a></li>
    <li><a href="docs/cuda_runtime/cudart_stream_ordered_memory_allocator.md">Stream Ordered Memory Allocator</a></li>
    <li><a href="docs/cuda_runtime/cudart_unified_addressing.md">Unified Addressing</a></li>
    <li><a href="docs/cuda_runtime/cudart_peer_device_memory_access.md">Peer Device Memory Access</a></li>
    <li><a href="docs/cuda_runtime/cudart_opengl_interoperability.md">OpenGL Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_direct3d_9_interoperability.md">Direct3D 9 Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_direct3d_10_interoperability.md">Direct3D 10 Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_direct3d_11_interoperability.md">Direct3D 11 Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_vdpau_interoperability.md">VDPAU Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_egl_interoperability.md">EGL Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_graphics_interoperability.md">Graphics Interoperability</a></li>
    <li><a href="docs/cuda_runtime/cudart_texture_object_management.md">Texture Object Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_surface_object_management.md">Surface Object Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_version_management.md">Version Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_graph_management.md">Graph Management</a></li>
    <li><a href="docs/cuda_runtime/cudart_driver_entry_point_access.md">Driver Entry Point Access</a></li>
    <li><a href="docs/cuda_runtime/cudart_profiler_control.md">Profiler Control</a></li>
</ul>


<h3>CUDA Driver APIs</h3>
<p>
Reference: Official CUDA Documentation for Driver APIs: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/index.html
<p>
Below lists supported status of each API under different categories.
<ul>
    <li><a href="docs/cuda_driver/cudadv_context_management.md">Context Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_device_management.md">Device Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_driver_entry_point_access.md">Driver Entry Point Access</a></li>
    <li><a href="docs/cuda_driver/cudadv_egl_interoperability.md">EGL Interoperability</a></li>
    <li><a href="docs/cuda_driver/cudadv_error_handling.md">Error Handling</a></li>
    <li><a href="docs/cuda_driver/cudadv_event_management.md">Event Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_execution_control.md">Execution Control</a></li>
    <li><a href="docs/cuda_driver/cudadv_external_resource_interoperability.md">External Resource Interoperability</a></li>
    <li><a href="docs/cuda_driver/cudadv_graph_management.md">Graph Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_graphics_interoperability.md">Graphics Interoperability</a></li>
    <li><a href="docs/cuda_driver/cudadv_initialization.md">Initialization</a></li>
    <li><a href="docs/cuda_driver/cudadv_memory_management.md">Memory Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_module_management.md">Module Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_occupancy.md">Occupancy</a></li>
    <li><a href="docs/cuda_driver/cudadv_opengl_interoperability.md">OpenGL Interoperability</a></li>
    <li><a href="docs/cuda_driver/cudadv_peer_context_memory_access.md">Peer Context Memory Access</a></li>
    <li><a href="docs/cuda_driver/cudadv_primary_context_management.md">Primary Context Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_profiler_control.md">Profiler Control</a></li>
    <li><a href="docs/cuda_driver/cudadv_stream_management.md">Stream Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_stream_memory_operations.md">Stream Memory Operations</a></li>
    <li><a href="docs/cuda_driver/cudadv_stream_ordered_memory_allocator.md">Stream Ordered Memory Allocator</a></li>
    <li><a href="docs/cuda_driver/cudadv_surface_object_management.md">Surface Object Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_texture_object_management.md">Texture Object Management</a></li>
    <li><a href="docs/cuda_driver/cudadv_unified_addressing.md">Unified Addressing</a></li>
    <li><a href="docs/cuda_driver/cudadv_vdpau_interoperability.md">VDPAU Interoperability</a></li>
</ul>


<h3>cuBLAS APIs</h3>
<p>
Reference: Official CUDA Documentation for cuBLAS APIs: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#using-the-cublas-api
<p>
Below lists supported status of each API under different categories.
<ul>
    <li><a href="docs/cublas/cublas_helper_function.md">Helper Function</a></li>
    <li><a href="docs/cublas/cublas_blas_level_1.md">BLAS Level 1</a></li>
    <li><a href="docs/cublas/cublas_blas_level_2.md">BLAS Level 2</a></li>
    <li><a href="docs/cublas/cublas_blas_level_3.md">BLAS Level 3</a></li>
    <li><a href="docs/cublas/cublas_blas_like_extension.md">BLAS-like Extension</a></li>
</ul>
 

<h3>cuBLASLt APIs</h3>
<p>
Reference: Official CUDA Documentation for cuBLASLt APIs: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#using-the-cublasLt-api
<p>
Below lists supported status of each API under different categories.
<ul>
    <li><a href="docs/cublaslt/cublaslt.md">cuBLASLt APIs</a></li>
</ul>


<h3>cuBLASXt APIs</h3>
<p>
Reference: Official CUDA Documentation for cuBLASXt APIs: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#using-the-cublasXt-api
<p>
Below lists supported status of each API under different categories.
<ul>
    <li><a href="docs/cublasxt/cublasxt_helper_function.md">Helper Function</a></li>
    <li><a href="docs/cublasxt/cublasxt_math_function.md">Math Function</a></li>
</ul>


<h3>cuDNN 8 APIs</h3>
<p>
Reference: Official CUDA Documentation for cuDNN 8 APIs: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/api/index.html. Note that cuDNN 8.0 is supported for CUDA 11.3.
<p>
Below lists supported status of each API under different categories.
<ul>
    <li><a href="docs/cudnn8/cudnn8_cudnn_ops_infer.md">cuDNN Ops Infer</a></li>
    <li><a href="docs/cudnn8/cudnn8_cudnn_ops_train.md">cuDNN Ops Train</a></li>
    <li><a href="docs/cudnn8/cudnn8_cudnn_cnn_infer.md">cuDNN CNN Inference</a></li>
    <li><a href="docs/cudnn8/cudnn8_cudnn_cnn_train.md">cuDNN CNN Train</a></li>
    <li><a href="docs/cudnn8/cudnn8_cudnn_adv_infer.md">cuDNN Advanced Inference</a></li>
    <li><a href="docs/cudnn8/cudnn8_cudnn_adv_train.md">cuDNN Advanced Train</a></li>
    <li><a href="docs/cudnn8/cudnn8_cudnn_backend.md">cuDNN Backend</a></li>
</ul>


<h3>NVML APIs</h3>
<p>
Reference: Official CUDA Documentation for NVML APIs: https://docs.nvidia.com/deploy/nvml-api/modules.html#modules
<p>
TODO: Add NVML APIs support status.


<h3>nvRTC APIs</h3>
<p>
Reference: Official CUDA Documentation for nvRTC APIs: https://docs.nvidia.com/cuda/archive/11.3.0/nvrtc/index.html
<p>
TODO: Add nvRTC APIs support status.


<h3>NCCL APIs</h3>
<p>
Reference: Official CUDA Documentation for NCCL APIs: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_299/user-guide/docs/index.html. Note that NCCL 2.99 is supported for CUDA 11.3.
<p>
TODO: Add NCCL APIs support status.
