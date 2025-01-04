<h2>PhOS Support: CUDA 11.3 - cuBLAS BLAS-like Extension Functions (0/53)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#blas-like-extension

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc)</code><br>
Performs matrix addition (single precision).
</td>
</tr>
<tr>
<td>1400</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc)</code><br>
Performs matrix addition (double precision).
</td>
</tr>
<tr>
<td>1401</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc)</code><br>
Performs matrix addition (complex single precision).
</td>
</tr>
<tr>
<td>1402</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc)</code><br>
Performs matrix addition (complex double precision).
</td>
</tr>
<tr>
<td>1403</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t side, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc)</code><br>
Performs diagonal matrix multiplication (single precision).
</td>
</tr>
<tr>
<td>1404</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t side, int m, int n, const double *A, int lda, const double *x, int incx, double *C, int ldc)</code><br>
Performs diagonal matrix multiplication (double precision).
</td>
</tr>
<tr>
<td>1405</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t side, int m, int n, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex *C, int ldc)</code><br>
Performs diagonal matrix multiplication (complex single precision).
</td>
</tr>
<tr>
<td>1406</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t side, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc)</code><br>
Performs diagonal matrix multiplication (complex double precision).
</td>
</tr>
<tr>
<td>1407</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize)</code><br>
Performs LU factorization of a batch of matrices (single precision).
</td>
</tr>
<tr>
<td>1408</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize)</code><br>
Performs LU factorization of a batch of matrices (double precision).
</td>
</tr>
<tr>
<td>1409</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize)</code><br>
Performs LU factorization of a batch of matrices (complex single precision).
</td>
</tr>
<tr>
<td>1410</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize)</code><br>
Performs LU factorization of a batch of matrices (complex double precision).
</td>
</tr>
<tr>
<td>1411</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *Aarray[], int lda, const int *PivotArray, float *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear systems using LU factorization (single precision).
</td>
</tr>
<tr>
<td>1412</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *Aarray[], int lda, const int *PivotArray, double *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear systems using LU factorization (double precision).
</td>
</tr>
<tr>
<td>1413</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex *Aarray[], int lda, const int *PivotArray, cuComplex *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear systems using LU factorization (complex single precision).
</td>
</tr>
<tr>
<td>1414</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *Aarray[], int lda, const int *PivotArray, cuDoubleComplex *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear systems using LU factorization (complex double precision).
</td>
</tr>
<tr>
<td>1415</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float *Aarray[], int lda, const int *PivotArray, float *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices using LU factorization (single precision).
</td>
</tr>
<tr>
<td>1416</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double *Aarray[], int lda, const int *PivotArray, double *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices using LU factorization (double precision).
</td>
</tr>
<tr>
<td>1417</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex *Aarray[], int lda, const int *PivotArray, cuComplex *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices using LU factorization (complex single precision).
</td>
</tr>
<tr>
<td>1418</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex *Aarray[], int lda, const int *PivotArray, cuDoubleComplex *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices using LU factorization (complex double precision).
</td>
</tr>
<tr>
<td>1419</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n, const float *Aarray[], int lda, float *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices (single precision).
</td>
</tr>
<tr>
<td>1420</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n, const double *Aarray[], int lda, double *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices (double precision).
</td>
</tr>
<tr>
<td>1421</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex *Aarray[], int lda, cuComplex *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices (complex single precision).
</td>
</tr>
<tr>
<td>1422</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex *Aarray[], int lda, cuDoubleComplex *Carray[], int ldc, int *infoArray, int batchSize)</code><br>
Computes the inverse of a batch of matrices (complex double precision).
</td>
</tr>
<tr>
<td>1423</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float *Aarray[], int lda, float *TauArray, int *infoArray, int batchSize)</code><br>
Computes the QR factorization of a batch of matrices (single precision).
</td>
</tr>
<tr>
<td>1424</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double *Aarray[], int lda, double *TauArray, int *infoArray, int batchSize)</code><br>
Computes the QR factorization of a batch of matrices (double precision).
</td>
</tr>
<tr>
<td>1425</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex *Aarray[], int lda, cuComplex *TauArray, int *infoArray, int batchSize)</code><br>
Computes the QR factorization of a batch of matrices (complex single precision).
</td>
</tr>
<tr>
<td>1426</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex *Aarray[], int lda, cuDoubleComplex *TauArray, int *infoArray, int batchSize)</code><br>
Computes the QR factorization of a batch of matrices (complex double precision).
</td>
</tr>
<tr>
<td>1427</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *Aarray[], int lda, float *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear least squares problems (single precision).
</td>
</tr>
<tr>
<td>1428</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *Aarray[], int lda, double *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear least squares problems (double precision).
</td>
</tr>
<tr>
<td>1429</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex *Aarray[], int lda, cuComplex *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear least squares problems (complex single precision).
</td>
</tr>
<tr>
<td>1430</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex *Aarray[], int lda, cuDoubleComplex *Barray[], int ldb, int *infoArray, int batchSize)</code><br>
Solves a batch of linear least squares problems (complex double precision).
</td>
</tr>
<tr>
<td>1431</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStpttr(cublasHandle_t handle, int n, const float *AP, float *A, int lda)</code><br>
Converts a triangular matrix from packed format to standard format (single precision).
</td>
</tr>
<tr>
<td>1432</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtpttr(cublasHandle_t handle, int n, const double *AP, double *A, int lda)</code><br>
Converts a triangular matrix from packed format to standard format (double precision).
</td>
</tr>
<tr>
<td>1433</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtpttr(cublasHandle_t handle, int n, const cuComplex *AP, cuComplex *A, int lda)</code><br>
Converts a triangular matrix from packed format to standard format (complex single precision).
</td>
</tr>
<tr>
<td>1434</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtpttr(cublasHandle_t handle, int n, const cuDoubleComplex *AP, cuDoubleComplex *A, int lda)</code><br>
Converts a triangular matrix from packed format to standard format (complex double precision).
</td>
</tr>
<tr>
<td>1435</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStrttp(cublasHandle_t handle, int n, const float *A, int lda, float *AP)</code><br>
Converts a triangular matrix from standard format to packed format (single precision).
</td>
</tr>
<tr>
<td>1436</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtrttp(cublasHandle_t handle, int n, const double *A, int lda, double *AP)</code><br>
Converts a triangular matrix from standard format to packed format (double precision).
</td>
</tr>
<tr>
<td>1437</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtrttp(cublasHandle_t handle, int n, const cuComplex *A, int lda, cuComplex *AP)</code><br>
Converts a triangular matrix from standard format to packed format (complex single precision).
</td>
</tr>
<tr>
<td>1438</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtrttp(cublasHandle_t handle, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *AP)</code><br>
Converts a triangular matrix from standard format to packed format (complex double precision).
</td>
</tr>
<tr>
<td>1439</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const float *beta, void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo)</code><br>
Performs matrix-matrix multiplication with extended precision (single precision).
</td>
</tr>
<tr>
<td>1440</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const void *beta, void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo)</code><br>
Performs matrix-matrix multiplication with extended precision.
</td>
</tr>
<tr>
<td>1441</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *Aarray[], cudaDataType Atype, int lda, const void *Barray[], cudaDataType Btype, int ldb, const void *beta, void *Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo)</code><br>
Performs batched matrix-matrix multiplication with extended precision.
</td>
</tr>
<tr>
<td>1442</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb, long long int strideB, const void *beta, void *C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo)</code><br>
Performs strided batched matrix-matrix multiplication with extended precision.
</td>
</tr>
<tr>
<td>1443</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType)</code><br>
Performs symmetric rank-k update with extended precision (complex single precision).
</td>
</tr>
<tr>
<td>1444</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType)</code><br>
Performs symmetric rank-k update using 3m algorithm with extended precision (complex single precision).
</td>
</tr>
<tr>
<td>1445</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType)</code><br>
Performs Hermitian rank-k update with extended precision (complex single precision).
</td>
</tr>
<tr>
<td>1446</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType)</code><br>
Performs Hermitian rank-k update using 3m algorithm with extended precision (complex single precision).
</td>
</tr>
<tr>
<td>1447</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void *x, cudaDataType xtype, int incx, void *result, cudaDataType resultType, cublasComputeType_t computeType)</code><br>
Computes the Euclidean norm of a vector with extended precision.
</td>
</tr>
<tr>
<td>1448</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphatype, const void *x, cudaDataType xtype, int incx, void *y, cudaDataType ytype, int incy, cublasComputeType_t computeType)</code><br>
Performs vector addition with extended precision.
</td>
</tr>
<tr>
<td>1449</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, cudaDataType xtype, int incx, const void *y, cudaDataType ytype, int incy, void *result, cudaDataType resultType, cublasComputeType_t computeType)</code><br>
Computes the dot product of two vectors with extended precision.
</td>
</tr>
<tr>
<td>1450</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void *x, cudaDataType xtype, int incx, void *y, cudaDataType ytype, int incy, const void *c, cudaDataType ctype, const void *s, cudaDataType stype, cublasComputeType_t computeType)</code><br>
Applies a Givens rotation to a vector with extended precision.
</td>
</tr>
<tr>
<td>1451</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphatype, void *x, cudaDataType xtype, int incx, cublasComputeType_t computeType)</code><br>
Scales a vector by a scalar with extended precision.
</td>
</tr>
<tr>
<td>1452</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
