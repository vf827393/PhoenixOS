<h2>PhOS Support: CUDA 11.3 - cuBLAS Helper Functions (0/24)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#cublas-helper-function-reference

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCreate(cublasHandle_t *handle)</code><br>
Initializes the cuBLAS library and creates a handle to an opaque structure holding the cuBLAS library context.
</td>
</tr>
<tr>
<td>1100</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDestroy(cublasHandle_t handle)</code><br>
Releases hardware resources used by the cuBLAS library.
</td>
</tr>
<tr>
<td>1101</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version)</code><br>
Returns the version number of the cuBLAS library.
</td>
</tr>
<tr>
<td>1102</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value)</code><br>
Returns the value of the requested property.
</td>
</tr>
<tr>
<td>1103</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)</code><br>
Sets the cuBLAS library stream for subsequent calls.
</td>
</tr>
<tr>
<td>1104</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes)</code><br>
Sets the cuBLAS library workspace to a user-owned device buffer.
</td>
</tr>
<tr>
<td>1105</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId)</code><br>
Gets the cuBLAS library stream being used.
</td>
</tr>
<tr>
<td>1106</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode)</code><br>
Obtains the pointer mode used by the cuBLAS library.
</td>
</tr>
<tr>
<td>1107</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode)</code><br>
Sets the pointer mode used by the cuBLAS library.
</td>
</tr>
<tr>
<td>1108</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)</code><br>
Copies n elements from a vector x in host memory to a vector y in GPU memory.
</td>
</tr>
<tr>
<td>1109</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)</code><br>
Copies n elements from a vector x in GPU memory to a vector y in host memory.
</td>
</tr>
<tr>
<td>1110</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)</code><br>
Copies a tile of rows x cols elements from a matrix A in host memory to a matrix B in GPU memory.
</td>
</tr>
<tr>
<td>1111</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)</code><br>
Copies a tile of rows x cols elements from a matrix A in GPU memory to a matrix B in host memory.
</td>
</tr>
<tr>
<td>1112</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream)</code><br>
Asynchronously copies n elements from a vector in host memory to a vector in GPU memory.
</td>
</tr>
<tr>
<td>1113</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream)</code><br>
Asynchronously copies n elements from a vector in GPU memory to a vector in host memory.
</td>
</tr>
<tr>
<td>1114</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)</code><br>
Asynchronously copies a tile of rows x cols elements from a matrix A in host memory to a matrix B in GPU memory.
</td>
</tr>
<tr>
<td>1115</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)</code><br>
Asynchronously copies a tile of rows x cols elements from a matrix A in GPU memory to a matrix B in host memory.
</td>
</tr>
<tr>
<td>1116</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode)</code><br>
Sets the atomics mode for cuBLAS routines.
</td>
</tr>
<tr>
<td>1117</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode)</code><br>
Gets the atomics mode for cuBLAS routines.
</td>
</tr>
<tr>
<td>1118</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)</code><br>
Sets the math mode for cuBLAS routines.
</td>
</tr>
<tr>
<td>1119</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode)</code><br>
Gets the math mode for cuBLAS routines.
</td>
</tr>
<tr>
<td>1120</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLoggerConfigure(cublasHandle_t handle, cublasLoggerCallback_t callback, void *userData)</code><br>
Configures the logger for cuBLAS routines.
</td>
</tr>
<tr>
<td>1121</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGetLoggerCallback(cublasHandle_t handle, cublasLoggerCallback_t *callback, void **userData)</code><br>
Gets the logger callback for cuBLAS routines.
</td>
</tr>
<tr>
<td>1122</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSetLoggerCallback(cublasHandle_t handle, cublasLoggerCallback_t callback, void *userData)</code><br>
Sets the logger callback for cuBLAS routines.
</td>
</tr>
<tr>
<td>1123</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
