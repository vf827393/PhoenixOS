<h2>PhOS Support: CUDA Runtime API - Direct3D 10 Interoperability (0/3)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__D3D10.html#group__CUDART__D3D10

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D10GetDevice(int* device, IDXGIAdapter* pAdapter)</code><br>
Gets the device number for an adapter.
</td>
</tr>
<tr>
<td>440</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D10GetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, ID3D10Device* pD3D10Device, cudaD3D10DeviceList deviceList)</code><br>
Gets the CUDA devices corresponding to a Direct3D 10 device.
</td>
</tr>
<tr>
<td>441</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsD3D10RegisterResource(cudaGraphicsResource** resource, ID3D10Resource* pD3DResource, unsigned int flags)</code><br>
Registers a Direct3D 10 resource for access by CUDA.
</td>
</tr>
<tr>
<td>442</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
