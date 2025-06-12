<h2>PhOS Support: CUDA 11.3 - Driver APIs - Peer Context Memory Access (0/4)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__PEER__ACCESS.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxDisablePeerAccess ( CUcontext peerContext )</code><br>
Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
</td>
</tr>
<tr>
<td>1030</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuCtxEnablePeerAccess ( CUcontext peerContext, unsigned int  Flags )</code><br>
Enables direct access to memory allocations in a peer context.
</td>
</tr>
<tr>
<td>1031</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceCanAccessPeer ( int* canAccessPeer, CUdevice dev, CUdevice peerDev )</code><br>
Queries if a device may directly access a peer device's memory.
</td>
</tr>
<tr>
<td>1032</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetP2PAttribute ( int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice )</code><br>
Queries attributes of the link between two devices.
</td>
</tr>
<tr>
<td>1033</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
