<h2>PhOS Support: CUDA Runtime API - Direct3D 9 Interoperability (0/5)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__D3D9.html#group__CUDART__D3D9

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D9GetDevice(int* device, const char* pszAdapterName)</code><br>
Gets the device number for an adapter.
</td>
</tr>
<tr>
<td>430</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D9GetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, IDirect3DDevice9* pD3D9Device, cudaD3D9DeviceList deviceList)</code><br>
Gets the CUDA devices corresponding to a Direct3D 9 device.
</td>
</tr>
<tr>
<td>431</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D9GetDirect3DDevice(IDirect3DDevice9** ppD3D9Device)</code><br>
Gets the Direct3D device against which the current CUDA context was created.
</td>
</tr>
<tr>
<td>432</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaD3D9SetDirect3DDevice(IDirect3DDevice9* pD3D9Device, int device = -1)</code><br>
Sets the Direct3D 9 device to use for interoperability with a CUDA device.
</td>
</tr>
<tr>
<td>433</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsD3D9RegisterResource(cudaGraphicsResource** resource, IDirect3DResource9* pD3DResource, unsigned int flags)</code><br>
Registers a Direct3D 9 resource for access by CUDA.
</td>
</tr>
<tr>
<td>434</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
