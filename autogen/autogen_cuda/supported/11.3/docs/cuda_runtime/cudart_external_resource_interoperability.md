<h2>PhOS Support: CUDA 11.3 - Runtime APIs - External Resource Interoperability (0/8)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDestroyExternalMemory ( cudaExternalMemory_t extMem )</code><br>
Destroys an external memory object.
</td>
</tr>
<tr>
<td>220</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDestroyExternalSemaphore ( cudaExternalSemaphore_t extSem )</code><br>
Destroys an external semaphore.
</td>
</tr>
<tr>
<td>221</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaExternalMemoryGetMappedBuffer ( void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc )</code><br>
Maps a buffer onto an imported memory object.
</td>
</tr>
<tr>
<td>222</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaExternalMemoryGetMappedMipmappedArray ( cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc )</code><br>
Maps a CUDA mipmapped array onto an external memory object.
</td>
</tr>
<tr>
<td>223</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaImportExternalMemory ( cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc )</code><br>
Imports an external memory object.
</td>
</tr>
<tr>
<td>224</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaImportExternalSemaphore ( cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc )</code><br>
Imports an external semaphore.
</td>
</tr>
<tr>
<td>225</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaSignalExternalSemaphoresAsync ( const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int  numExtSems, cudaStream_t stream = 0 )</code><br>
Signals a set of external semaphore objects.
</td>
</tr>
<tr>
<td>226</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaWaitExternalSemaphoresAsync ( const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int  numExtSems, cudaStream_t stream = 0 )</code><br>
Waits on a set of external semaphore objects.
</td>
</tr>
<tr>
<td>227</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
