<h2>PhOS Support: CUDA 11.3 - Driver APIs - Surface Object Management (0/3)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__SURFOBJECT.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuSurfObjectCreate ( CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc )</code><br>
Creates a surface object.
</td>
</tr>
<tr>
<td>1020</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuSurfObjectDestroy ( CUsurfObject surfObject )</code><br>
Destroys a surface object.
</td>
</tr>
<tr>
<td>1021</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuSurfObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject )</code><br>
Returns a surface object's resource descriptor.
</td>
</tr>
<tr>
<td>1022</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
