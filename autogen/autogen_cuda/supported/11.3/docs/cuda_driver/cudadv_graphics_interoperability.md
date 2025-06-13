<h2>PhOS Support: CUDA 11.3 - Driver APIs - Graphics Interoperability (0/7)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__GRAPHICS.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsMapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream )</code><br>
Map graphics resources for access by CUDA.
</td>
</tr>
<tr>
<td>1040</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsResourceGetMappedMipmappedArray ( CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource )</code><br>
Get a mipmapped array through which to access a mapped graphics resource.
</td>
</tr>
<tr>
<td>1041</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsResourceGetMappedPointer ( CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource )</code><br>
Get a device pointer through which to access a mapped graphics resource.
</td>
</tr>
<tr>
<td>1042</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsResourceSetMapFlags ( CUgraphicsResource resource, unsigned int  flags )</code><br>
Set usage flags for mapping a graphics resource.
</td>
</tr>
<tr>
<td>1043</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsSubResourceGetMappedArray ( CUarray* pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel )</code><br>
Get an array through which to access a subresource of a mapped graphics resource.
</td>
</tr>
<tr>
<td>1044</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsUnmapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream )</code><br>
Unmap graphics resources.
</td>
</tr>
<tr>
<td>1045</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsUnregisterResource ( CUgraphicsResource resource )</code><br>
Unregisters a graphics resource for access by CUDA.
</td>
</tr>
<tr>
<td>1046</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
