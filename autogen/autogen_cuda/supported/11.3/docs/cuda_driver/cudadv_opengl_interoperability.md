<h2>PhOS Support: CUDA 11.3 - Driver APIs - OpenGL Interoperability (0/4)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__OPENGL.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGLGetDevices ( unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int  cudaDeviceCount, CUGLDeviceList deviceList )</code><br>
Gets the CUDA devices associated with the current OpenGL context.
</td>
</tr>
<tr>
<td>1070</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsGLRegisterBuffer ( CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int  Flags )</code><br>
Registers an OpenGL buffer object.
</td>
</tr>
<tr>
<td>1071</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsGLRegisterImage ( CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int  Flags )</code><br>
Register an OpenGL texture or renderbuffer object.
</td>
</tr>
<tr>
<td>1072</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuWGLGetDevice ( CUdevice* pDevice, HGPUNV hGpu )</code><br>
Gets the CUDA device associated with hGpu.
</td>
</tr>
<tr>
<td>1073</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
