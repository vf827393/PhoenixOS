<h2>PhOS Support: CUDA Runtime API - VDPAU Interoperability (0/4)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__VDPAU.html#group__CUDART__VDPAU

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, VdpOutputSurface vdpSurface, unsigned int flags)</code><br>
Registers a VdpOutputSurface object for access by CUDA.
</td>
</tr>
<tr>
<td>460</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, VdpVideoSurface vdpSurface, unsigned int flags)</code><br>
Registers a VdpVideoSurface object for access by CUDA.
</td>
</tr>
<tr>
<td>461</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaVDPAUGetDevice(int* device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress)</code><br>
Gets the CUDA device associated with a VdpDevice.
</td>
</tr>
<tr>
<td>462</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress)</code><br>
Sets a CUDA device to use VDPAU interoperability.
</td>
</tr>
<tr>
<td>463</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
