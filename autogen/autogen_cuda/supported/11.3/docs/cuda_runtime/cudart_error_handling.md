<h2>PhOS Support: CUDA 11.3 - Runtime APIs - Error Handling (3/4)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>const char* cudaGetErrorName ( cudaError_t error )</code><br>
Returns the string representation of an error code enum name.
</td>
</tr>
<tr>
<td>150</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr style="background-color:green;">
<td colspan=3>
<code>const char* cudaGetErrorString ( cudaError_t error )</code><br>
Returns the description string for an error code.
</td>
</tr>
<tr>
<td>151</td>
<td>☑</td>
<td>✗</td>
</tr>

<tr style="background-color:green;">
<td colspan=3>
<code>cudaError_t cudaGetLastError ( void )</code><br>
Returns the last error from a runtime call.
</td>
</tr>
<tr>
<td>152</td>
<td>☑</td>
<td>✗</td>
</tr>

<tr style="background-color:green;">
<td colspan=3>
<code>cudaError_t cudaPeekAtLastError ( void )</code><br>
Returns the last error from a runtime call.
</td>
</tr>
<tr>
<td>153</td>
<td>☑</td>
<td>✗</td>
</tr>
</table>
