<h2>PhOS Support: CUDA Runtime API - Direct3D 11 Interoperability (0/3)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__D3D11.html#group__CUDART__D3D11

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D11GetDevice(int* device, IDXGIAdapter* pAdapter)</code><br>
Gets the device number for an adapter.
</td>
</tr>
<tr>
<td>450</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D11GetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, ID3D11Device* pD3D11Device, cudaD3D11DeviceList deviceList)</code><br>
Gets the CUDA devices corresponding to a Direct3D 11 device.
</td>
</tr>
<tr>
<td>451</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsD3D11RegisterResource(cudaGraphicsResource** resource, ID3D11Resource* pD3DResource, unsigned int flags)</code><br>
Registers a Direct3D 11 resource for access by CUDA.
</td>
</tr>
<tr>
<td>452</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
