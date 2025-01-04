<h2>PhOS Support: CUDA 11.3 - Driver APIs - Context Management (0/16)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxCreate ( CUcontext* pctx, unsigned int flags, CUdevice dev )</code><br>
Create a CUDA context.
</td>
</tr>
<tr>
<td>680</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxDestroy ( CUcontext ctx )</code><br>
Destroy a CUDA context.
</td>
</tr>
<tr>
<td>681</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version )</code><br>
Gets the context's API version.
</td>
</tr>
<tr>
<td>682</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig )</code><br>
Returns the preferred cache configuration for the current context.
</td>
</tr>
<tr>
<td>683</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetCurrent ( CUcontext* pctx )</code><br>
Returns the CUDA context bound to the calling CPU thread.
</td>
</tr>
<tr>
<td>684</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetDevice ( CUdevice* device )</code><br>
Returns the device ID for the current context.
</td>
</tr>
<tr>
<td>685</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetFlags ( unsigned int* flags )</code><br>
Returns the flags for the current context.
</td>
</tr>
<tr>
<td>686</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetLimit ( size_t* pvalue, CUlimit limit )</code><br>
Returns resource limits.
</td>
</tr>
<tr>
<td>687</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig )</code><br>
Returns the current shared memory configuration for the current context.
</td>
</tr>
<tr>
<td>688</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority )</code><br>
Returns numerical values that correspond to the least and greatest stream priorities.
</td>
</tr>
<tr>
<td>689</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxPopCurrent ( CUcontext* pctx )</code><br>
Pops the current CUDA context from the current CPU thread.
</td>
</tr>
<tr>
<td>690</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxPushCurrent ( CUcontext ctx )</code><br>
Pushes a context on the current CPU thread.
</td>
</tr>
<tr>
<td>691</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxResetPersistingL2Cache ( void )</code><br>
Resets all persisting lines in cache to normal status.
</td>
</tr>
<tr>
<td>692</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxSetCacheConfig ( CUfunc_cache config )</code><br>
Sets the preferred cache configuration for the current context.
</td>
</tr>
<tr>
<td>693</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxSetCurrent ( CUcontext ctx )</code><br>
Binds the specified CUDA context to the calling CPU thread.
</td>
</tr>
<tr>
<td>694</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxSetLimit ( CUlimit limit, size_t value )</code><br>
Set resource limits.
</td>
</tr>
<tr>
<td>695</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config )</code><br>
Sets the shared memory configuration for the current context.
</td>
</tr>
<tr>
<td>696</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxSynchronize ( void )</code><br>
Block for a context's tasks to complete.
</td>
</tr>
<tr>
<td>697</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
