<h2>PhOS Support: CUDA 11.3 - Driver APIs - VDPAU Interoperability (0/4)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__VDPAU.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsVDPAURegisterOutputSurface ( CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int  flags )</code><br>
Registers a VDPAU VdpOutputSurface object.
</td>
</tr>
<tr>
<td>1080</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsVDPAURegisterVideoSurface ( CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int  flags )</code><br>
Registers a VDPAU VdpVideoSurface object.
</td>
</tr>
<tr>
<td>1081</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuVDPAUCtxCreate ( CUcontext* pCtx, unsigned int  flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress )</code><br>
Create a CUDA context for interoperability with VDPAU.
</td>
</tr>
<tr>
<td>1082</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuVDPAUGetDevice ( CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress )</code><br>
Gets the CUDA device associated with a VDPAU device.
</td>
</tr>
<tr>
<td>1083</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
