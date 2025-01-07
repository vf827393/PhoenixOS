<h1>PhOS Support for CUDA 11.3</h1>

<p>
PhOS is supporting popular SDKs on nVIDIA CUDA platforms (e.g., CUDA Runtime, CUDA Driver, cuBLAS, cuBLASLt, cuBLASXt, cuDNN 8, NVML, nvRTC, and NCCL, etc.). This directory contains `yaml` files which describe descriptors of CUDA 11.3 APIs for autogenerating processing logic of PhOS parser and worker functions.

<p>
You can check the supported status in PhOS of each API under different categories by clicking the link below.


<h2>CUDA Runtime APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/index.html
<p>
Below lists supported status of each API under different categories.
<ul>
    <li>Device Management <a href="docs/cuda_runtime/cudart_device_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_device_management.yaml">[yaml]</a></li>
    <li>Error Handling <a href="docs/cuda_runtime/cudart_error_handling.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_error_handling.yaml">[yaml]</a></li>
    <li>Stream Management <a href="docs/cuda_runtime/cudart_stream_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_stream_management.yaml">[yaml]</a></li>
    <li>Event Management <a href="docs/cuda_runtime/cudart_event_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_event_management.yaml">[yaml]</a></li>
    <li>External Resource Interoperability <a href="docs/cuda_runtime/cudart_external_resource_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_external_resource_interoperability.yaml">[yaml]</a></li>
    <li>Execution Control <a href="docs/cuda_runtime/cudart_execution_control.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_execution_control.yaml">[yaml]</a></li>
    <li>Memory Management <a href="docs/cuda_runtime/cudart_memory_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_memory_management.yaml">[yaml]</a></li>
    <li>Occupancy <a href="docs/cuda_runtime/cudart_occupancy.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_occupancy.yaml">[yaml]</a></li>
    <li>Stream Ordered Memory Allocator <a href="docs/cuda_runtime/cudart_stream_ordered_memory_allocator.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_stream_ordered_memory_allocator.yaml">[yaml]</a></li>
    <li>Unified Addressing <a href="docs/cuda_runtime/cudart_unified_addressing.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_unified_addressing.yaml">[yaml]</a></li>
    <li>Peer Device Memory Access <a href="docs/cuda_runtime/cudart_peer_device_memory_access.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_peer_device_memory_access.yaml">[yaml]</a></li>
    <li>OpenGL Interoperability <a href="docs/cuda_runtime/cudart_opengl_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_opengl_interoperability.yaml">[yaml]</a></li>
    <li>Direct3D 9 Interoperability <a href="docs/cuda_runtime/cudart_direct3d_9_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_direct3d_9_interoperability.yaml">[yaml]</a></li>
    <li>Direct3D 10 Interoperability <a href="docs/cuda_runtime/cudart_direct3d_10_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_direct3d_10_interoperability.yaml">[yaml]</a></li>
    <li>Direct3D 11 Interoperability <a href="docs/cuda_runtime/cudart_direct3d_11_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_direct3d_11_interoperability.yaml">[yaml]</a></li>
    <li>VDPAU Interoperability <a href="docs/cuda_runtime/cudart_vdpau_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_vdpau_interoperability.yaml">[yaml]</a></li>
    <li>EGL Interoperability <a href="docs/cuda_runtime/cudart_egl_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_egl_interoperability.yaml">[yaml]</a></li>
    <li>Graphics Interoperability <a href="docs/cuda_runtime/cudart_graphics_interoperability.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_graphics_interoperability.yaml">[yaml]</a></li>
    <li>Texture Object Management <a href="docs/cuda_runtime/cudart_texture_object_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_texture_object_management.yaml">[yaml]</a></li>
    <li>Surface Object Management <a href="docs/cuda_runtime/cudart_surface_object_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_surface_object_management.yaml">[yaml]</a></li>
    <li>Version Management <a href="docs/cuda_runtime/cudart_version_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_version_management.yaml">[yaml]</a></li>
    <li>Graph Management <a href="docs/cuda_runtime/cudart_graph_management.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_graph_management.yaml">[yaml]</a></li>
    <li>Driver Entry Point Access <a href="docs/cuda_runtime/cudart_driver_entry_point_access.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_driver_entry_point_access.yaml">[yaml]</a></li>
    <li>Profiler Control <a href="docs/cuda_runtime/cudart_profiler_control.md">[doc]</a> <a href="yaml/cuda_runtime/cudart_profiler_control.yaml">[yaml]</a></li>
</ul>


<h2>CUDA Driver APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/index.html
<p>
Below lists supported status of each API under different categories.
<ul>
    <li>Context Management <a href="docs/cuda_driver/cudadv_context_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_context_management.yaml">[yaml]</a></li>
    <li>Device Management <a href="docs/cuda_driver/cudadv_device_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_device_management.yaml">[yaml]</a></li>
    <li>Driver Entry Point Access <a href="docs/cuda_driver/cudadv_driver_entry_point_access.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_driver_entry_point_access.yaml">[yaml]</a></li>
    <li>EGL Interoperability <a href="docs/cuda_driver/cudadv_egl_interoperability.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_egl_interoperability.yaml">[yaml]</a></li>
    <li>Error Handling <a href="docs/cuda_driver/cudadv_error_handling.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_error_handling.yaml">[yaml]</a></li>
    <li>Event Management <a href="docs/cuda_driver/cudadv_event_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_event_management.yaml">[yaml]</a></li>
    <li>Execution Control <a href="docs/cuda_driver/cudadv_execution_control.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_execution_control.yaml">[yaml]</a></li>
    <li>External Resource Interoperability <a href="docs/cuda_driver/cudadv_external_resource_interoperability.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_external_resource_interoperability.yaml">[yaml]</a></li>
    <li>Graph Management <a href="docs/cuda_driver/cudadv_graph_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_graph_management.yaml">[yaml]</a></li>
    <li>Graphics Interoperability <a href="docs/cuda_driver/cudadv_graphics_interoperability.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_graphics_interoperability.yaml">[yaml]</a></li>
    <li>Initialization <a href="docs/cuda_driver/cudadv_initialization.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_initialization.yaml">[yaml]</a></li>
    <li>Memory Management <a href="docs/cuda_driver/cudadv_memory_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_memory_management.yaml">[yaml]</a></li>
    <li>Module Management <a href="docs/cuda_driver/cudadv_module_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_module_management.yaml">[yaml]</a></li>
    <li>Occupancy <a href="docs/cuda_driver/cudadv_occupancy.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_occupancy.yaml">[yaml]</a></li>
    <li>OpenGL Interoperability <a href="docs/cuda_driver/cudadv_opengl_interoperability.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_opengl_interoperability.yaml">[yaml]</a></li>
    <li>Peer Context Memory Access <a href="docs/cuda_driver/cudadv_peer_context_memory_access.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_peer_context_memory_access.yaml">[yaml]</a></li>
    <li>Primary Context Management <a href="docs/cuda_driver/cudadv_primary_context_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_primary_context_management.yaml">[yaml]</a></li>
    <li>Profiler Control <a href="docs/cuda_driver/cudadv_profiler_control.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_profiler_control.yaml">[yaml]</a></li>
    <li>Stream Management <a href="docs/cuda_driver/cudadv_stream_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_stream_management.yaml">[yaml]</a></li>
    <li>Stream Memory Operations <a href="docs/cuda_driver/cudadv_stream_memory_operations.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_stream_memory_operations.yaml">[yaml]</a></li>
    <li>Stream Ordered Memory Allocator <a href="docs/cuda_driver/cudadv_stream_ordered_memory_allocator.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_stream_ordered_memory_allocator.yaml">[yaml]</a></li>
    <li>Surface Object Management <a href="docs/cuda_driver/cudadv_surface_object_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_surface_object_management.yaml">[yaml]</a></li>
    <li>Texture Object Management <a href="docs/cuda_driver/cudadv_texture_object_management.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_texture_object_management.yaml">[yaml]</a></li>
    <li>Unified Addressing <a href="docs/cuda_driver/cudadv_unified_addressing.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_unified_addressing.yaml">[yaml]</a></li>
    <li>VDPAU Interoperability <a href="docs/cuda_driver/cudadv_vdpau_interoperability.md">[doc]</a> <a href="yaml/cuda_driver/cudadv_vdpau_interoperability.yaml">[yaml]</a></li>
</ul>


<h2>cuBLAS APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#using-the-cublas-api
<p>
Below lists supported status of each API under different categories.
<ul>
    <li>Helper Function <a href="docs/cublas/cublas_helper_function.md">[doc]</a> <a href="yaml/cublas/helper_function.yaml">[yaml]</a></li>
    <li>BLAS Level 1 <a href="docs/cublas/cublas_blas_level_1.md">[doc]</a> <a href="yaml/cublas/blas_level_1.yaml">[yaml]</a></li>
    <li>BLAS Level 2 <a href="docs/cublas/cublas_blas_level_2.md">[doc]</a> <a href="yaml/cublas/blas_level_2.yaml">[yaml]</a></li>
    <li>BLAS Level 3 <a href="docs/cublas/cublas_blas_level_3.md">[doc]</a> <a href="yaml/cublas/blas_level_3.yaml">[yaml]</a></li>
    <li>BLAS-like Extension <a href="docs/cublas/cublas_blas_like_extension.md">[doc]</a> <a href="yaml/cublas/blas_like.yaml">[yaml]</a></li>
</ul>
 

<h2>cuBLASLt APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#using-the-cublasLt-api
<p>
Below lists supported status of each API under different categories.
<ul>
    <li>cuBLASLt APIs <a href="docs/cublaslt/cublaslt.md">[doc]</a> <a href="yaml/cublaslt/cublaslt.yaml">[yaml]</a></li>
</ul>


<h2>cuBLASXt APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#using-the-cublasXt-api
<p>
Below lists supported status of each API under different categories.
<ul>
    <li>Helper Function <a href="docs/cublasxt/cublasxt_helper_function.md">[doc]</a> <a href="yaml/cublasxt/cublasxt_helper_function.yaml">[yaml]</a></li>
    <li>Math Function <a href="docs/cublasxt/cublasxt_math_function.md">[doc]</a> <a href="yaml/cublasxt/cublasxt_math_function.yaml">[yaml]</a></li>
</ul>


<h2>cuDNN 8 APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/api/index.html. Note that cuDNN 8.0 is supported for CUDA 11.3.
<p>
Below lists supported status of each API under different categories.
<ul>
    <li>cuDNN Ops Infer <a href="docs/cudnn8/cudnn8_cudnn_ops_infer.md">[doc]</a> <a href="yaml/cudnn8/cudnn8_cudnn_ops_infer.yaml">[yaml]</a></li>
    <li>cuDNN Ops Train <a href="docs/cudnn8/cudnn8_cudnn_ops_train.md">[doc]</a> <a href="yaml/cudnn8/cudnn8_cudnn_ops_train.yaml">[yaml]</a></li>
    <li>cuDNN CNN Inference <a href="docs/cudnn8/cudnn8_cudnn_cnn_infer.md">[doc]</a> <a href="yaml/cudnn8/cudnn8_cudnn_cnn_infer.yaml">[yaml]</a></li>
    <li>cuDNN CNN Train <a href="docs/cudnn8/cudnn8_cudnn_cnn_train.md">[doc]</a> <a href="yaml/cudnn8/cudnn8_cudnn_cnn_train.yaml">[yaml]</a></li>
    <li>cuDNN Advanced Inference <a href="docs/cudnn8/cudnn8_cudnn_adv_infer.md">[doc]</a> <a href="yaml/cudnn8/cudnn8_cudnn_adv_infer.yaml">[yaml]</a></li>
    <li>cuDNN Advanced Train <a href="docs/cudnn8/cudnn8_cudnn_adv_train.md">[doc]</a> <a href="yaml/cudnn8/cudnn8_cudnn_adv_train.yaml">[yaml]</a></li>
    <li>cuDNN Backend <a href="docs/cudnn8/cudnn8_cudnn_backend.md">[doc]</a> <a href="yaml/cudnn8/cudnn8_cudnn_backend.yaml">[yaml]</a></li>
</ul>


<h2>NVML APIs</h2>
<p >
Reference: https://docs.nvidia.com/deploy/nvml-api/modules.html#modules
<p>
TODO: Add NVML APIs support status.


<h2>nvRTC APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/cuda/archive/11.3.0/nvrtc/index.html
<p>
TODO: Add nvRTC APIs support status.


<h2>NCCL APIs</h2>
<p style="color:grey;">
Reference: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_299/user-guide/docs/index.html. Note that NCCL 2.99 is supported for CUDA 11.3.
<p>
TODO: Add NCCL APIs support status.
