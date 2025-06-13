<h2>PhOS Support: CUDA Runtime API - Surface Object Management (0/3)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc)</code><br>
Creates a surface object.
</td>
</tr>
<tr>
<td>500</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)</code><br>
Destroys a surface object.
</td>
</tr>
<tr>
<td>501</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject)</code><br>
Returns a surface object's resource descriptor.
</td>
</tr>
<tr>
<td>502</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
