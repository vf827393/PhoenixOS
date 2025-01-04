<h2>PhOS Support: CUDA 11.3 - Driver APIs - Texture Object Management (0/5)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__TEXOBJECT.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuTexObjectCreate ( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc )</code><br>
Creates a texture object.
</td>
</tr>
<tr>
<td>1010</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuTexObjectDestroy ( CUtexObject texObject )</code><br>
Destroys a texture object.
</td>
</tr>
<tr>
<td>1011</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuTexObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject )</code><br>
Returns a texture object's resource descriptor.
</td>
</tr>
<tr>
<td>1012</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuTexObjectGetResourceViewDesc ( CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject )</code><br>
Returns a texture object's resource view descriptor.
</td>
</tr>
<tr>
<td>1013</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuTexObjectGetTextureDesc ( CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject )</code><br>
Returns a texture object's texture descriptor.
</td>
</tr>
<tr>
<td>1014</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
