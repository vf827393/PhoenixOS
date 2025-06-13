<h2>PhOS Support: CUDA Runtime API - OpenGL Interoperability (0/4)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList)</code><br>
Gets the CUDA devices associated with the current OpenGL context.
</td>
</tr>
<tr>
<td>420</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, GLuint buffer, unsigned int flags)</code><br>
Registers an OpenGL buffer object for access by CUDA.
</td>
</tr>
<tr>
<td>421</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags)</code><br>
Registers an OpenGL texture or renderbuffer object for access by CUDA.
</td>
</tr>
<tr>
<td>422</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaWGLGetDevice(int* device, HGPUNV hGpu)</code><br>
Gets the CUDA device associated with hGpu.
</td>
</tr>
<tr>
<td>423</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
