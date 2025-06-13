<h2>PhOS Support: CUDA Runtime API - Graphics Interoperability (0/7)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream = 0)</code><br>
Map graphics resources for access by CUDA.
</td>
</tr>
<tr>
<td>480</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource)</code><br>
Get a mipmapped array through which to access a mapped graphics resource.
</td>
</tr>
<tr>
<td>481</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource)</code><br>
Get a device pointer through which to access a mapped graphics resource.
</td>
</tr>
<tr>
<td>482</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)</code><br>
Set usage flags for mapping a graphics resource.
</td>
</tr>
<tr>
<td>483</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)</code><br>
Get an array through which to access a subresource of a mapped graphics resource.
</td>
</tr>
<tr>
<td>484</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream = 0)</code><br>
Unmap graphics resources.
</td>
</tr>
<tr>
<td>485</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)</code><br>
Unregisters a graphics resource for access by CUDA.
</td>
</tr>
<tr>
<td>486</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
