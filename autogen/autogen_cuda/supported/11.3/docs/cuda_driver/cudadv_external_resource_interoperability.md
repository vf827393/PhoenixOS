<h2>PhOS Support: CUDA 11.3 - Driver APIs - External Resource Interoperability (0/8)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDestroyExternalMemory ( CUexternalMemory extMem )</code><br>
Destroys an external memory object.
</td>
</tr>
<tr>
<td>890</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDestroyExternalSemaphore ( CUexternalSemaphore extSem )</code><br>
Destroys an external semaphore.
</td>
</tr>
<tr>
<td>891</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuExternalMemoryGetMappedBuffer ( CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc )</code><br>
Maps a buffer onto an imported memory object.
</td>
</tr>
<tr>
<td>892</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuExternalMemoryGetMappedMipmappedArray ( CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc )</code><br>
Maps a CUDA mipmapped array onto an external memory object.
</td>
</tr>
<tr>
<td>893</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuImportExternalMemory ( CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc )</code><br>
Imports an external memory object.
</td>
</tr>
<tr>
<td>894</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuImportExternalSemaphore ( CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc )</code><br>
Imports an external semaphore.
</td>
</tr>
<tr>
<td>895</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuSignalExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream )</code><br>
Signals a set of external semaphore objects.
</td>
</tr>
<tr>
<td>896</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuWaitExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream )</code><br>
Waits on a set of external semaphore objects.
</td>
</tr>
<tr>
<td>897</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
