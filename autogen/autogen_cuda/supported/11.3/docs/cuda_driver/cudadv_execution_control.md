<h2>PhOS Support: CUDA 11.3 - Driver APIs - Execution Control (0/9)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__EXEC.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuFuncGetAttribute ( int* pi, CUfunction_attribute attrib, CUfunction hfunc )</code><br>
Returns information about a function.
</td>
</tr>
<tr>
<td>910</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuFuncGetModule ( CUmodule* hmod, CUfunction hfunc )</code><br>
Returns a module handle.
</td>
</tr>
<tr>
<td>911</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuFuncSetAttribute ( CUfunction hfunc, CUfunction_attribute attrib, int  value )</code><br>
Sets information about a function.
</td>
</tr>
<tr>
<td>912</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config )</code><br>
Sets the preferred cache configuration for a device function.
</td>
</tr>
<tr>
<td>913</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuFuncSetSharedMemConfig ( CUfunction hfunc, CUsharedconfig config )</code><br>
Sets the shared memory configuration for a device function.
</td>
</tr>
<tr>
<td>914</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLaunchCooperativeKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams )</code><br>
Launches a CUDA function where thread blocks can cooperate and synchronize as they execute.
</td>
</tr>
<tr>
<td>915</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLaunchCooperativeKernelMultiDevice ( CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int  numDevices, unsigned int  flags )</code><br>
Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute.
</td>
</tr>
<tr>
<td>916</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLaunchHostFunc ( CUstream hStream, CUhostFn fn, void* userData )</code><br>
Enqueues a host function call in a stream.
</td>
</tr>
<tr>
<td>917</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )</code><br>
Launches a CUDA function.
</td>
</tr>
<tr>
<td>918</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
