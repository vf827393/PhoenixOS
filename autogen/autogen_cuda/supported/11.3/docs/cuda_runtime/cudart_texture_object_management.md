<h2>PhOS Support: CUDA Runtime API - Texture Object Management (0/7)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f)</code><br>
Returns a channel descriptor using the specified format.
</td>
</tr>
<tr>
<td>490</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc)</code><br>
Creates a texture object.
</td>
</tr>
<tr>
<td>491</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject)</code><br>
Destroys a texture object.
</td>
</tr>
<tr>
<td>492</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array)</code><br>
Get the channel descriptor of an array.
</td>
</tr>
<tr>
<td>493</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject)</code><br>
Returns a texture object's resource descriptor.
</td>
</tr>
<tr>
<td>494</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject)</code><br>
Returns a texture object's resource view descriptor.
</td>
</tr>
<tr>
<td>495</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject)</code><br>
Returns a texture object's texture descriptor.
</td>
</tr>
<tr>
<td>496</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
