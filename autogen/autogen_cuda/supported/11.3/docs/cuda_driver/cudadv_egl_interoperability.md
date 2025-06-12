<h2>PhOS Support: CUDA 11.3 - Driver APIs - EGL Interoperability (0/12)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__EGL.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamConsumerAcquireFrame ( CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int  timeout )</code><br>
Acquire an image frame from the EGLStream with CUDA as a consumer.
</td>
</tr>
<tr>
<td>1090</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamConsumerConnect ( CUeglStreamConnection* conn, EGLStreamKHR stream )</code><br>
Connect CUDA to EGLStream as a consumer.
</td>
</tr>
<tr>
<td>1091</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamConsumerConnectWithFlags ( CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int  flags )</code><br>
Connect CUDA to EGLStream as a consumer with given flags.
</td>
</tr>
<tr>
<td>1092</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamConsumerDisconnect ( CUeglStreamConnection* conn )</code><br>
Disconnect CUDA as a consumer to EGLStream.
</td>
</tr>
<tr>
<td>1093</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamConsumerReleaseFrame ( CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream )</code><br>
Releases the last frame acquired from the EGLStream.
</td>
</tr>
<tr>
<td>1094</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamProducerConnect ( CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height )</code><br>
Connect CUDA to EGLStream as a producer.
</td>
</tr>
<tr>
<td>1095</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamProducerDisconnect ( CUeglStreamConnection* conn )</code><br>
Disconnect CUDA as a producer to EGLStream.
</td>
</tr>
<tr>
<td>1096</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamProducerPresentFrame ( CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream )</code><br>
Present a CUDA eglFrame to the EGLStream with CUDA as a producer.
</td>
</tr>
<tr>
<td>1097</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEGLStreamProducerReturnFrame ( CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream )</code><br>
Return the CUDA eglFrame to the EGLStream released by the consumer.
</td>
</tr>
<tr>
<td>1098</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventCreateFromEGLSync ( CUevent* phEvent, EGLSyncKHR eglSync, unsigned int  flags )</code><br>
Creates an event from EGLSync object.
</td>
</tr>
<tr>
<td>1099</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsEGLRegisterImage ( CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int  flags )</code><br>
Registers an EGL image.
</td>
</tr>
<tr>
<td>1100</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphicsResourceGetMappedEglFrame ( CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int  index, unsigned int  mipLevel )</code><br>
Get an eglFrame through which to access a registered EGL graphics resource.
</td>
</tr>
<tr>
<td>1101</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
