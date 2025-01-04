# [Execution Control (0/12)](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION)

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

