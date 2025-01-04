<h2>PhOS Support: CUDA 11.3 - cuBLASXt Math Functions (0/20)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#cublasxt-api-ref

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSgemm(cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, size_t m, size_t n, size_t k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)</code><br>
Performs single-precision general matrix multiplication.
</td>
</tr>
<tr>
<td>1700</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDgemm(cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)</code><br>
Performs double-precision general matrix multiplication.
</td>
</tr>
<tr>
<td>1701</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCgemm(cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs single-precision complex general matrix multiplication.
</td>
</tr>
<tr>
<td>1702</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZgemm(cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs double-precision complex general matrix multiplication.
</td>
</tr>
<tr>
<td>1703</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtChemm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)</code><br>
Performs single-precision complex Hermitian matrix multiplication.
</td>
</tr>
<tr>
<td>1704</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZhemm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)</code><br>
Performs double-precision complex Hermitian matrix multiplication.
</td>
</tr>
<tr>
<td>1705</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSsymm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)</code><br>
Performs single-precision symmetric matrix multiplication.
</td>
</tr>
<tr>
<td>1706</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDsymm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)</code><br>
Performs double-precision symmetric matrix multiplication.
</td>
</tr>
<tr>
<td>1707</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCsymm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)</code><br>
Performs single-precision complex symmetric matrix multiplication.
</td>
</tr>
<tr>
<td>1708</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZsymm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)</code><br>
Performs double-precision complex symmetric matrix multiplication.
</td>
</tr>
<tr>
<td>1709</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc)</code><br>
Performs single-precision symmetric rank-k update.
</td>
</tr>
<tr>
<td>1710</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc)</code><br>
Performs double-precision symmetric rank-k update.
</td>
</tr>
<tr>
<td>1711</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs single-precision complex symmetric rank-k update.
</td>
</tr>
<tr>
<td>1712</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs double-precision complex symmetric rank-k update.
</td>
</tr>
<tr>
<td>1713</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)</code><br>
Performs single-precision symmetric rank-2k update.
</td>
</tr>
<tr>
<td>1714</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)</code><br>
Performs double-precision symmetric rank-2k update.
</td>
</tr>
<tr>
<td>1715</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)</code><br>
Performs single-precision complex symmetric rank-2k update.
</td>
</tr>
<tr>
<td>1716</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)</code><br>
Performs double-precision complex symmetric rank-2k update.
</td>
</tr>
<tr>
<td>1717</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCherk(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc)</code><br>
Performs single-precision complex Hermitian rank-k update.
</td>
</tr>
<tr>
<td>1718</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZherk(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs double-precision complex Hermitian rank-k update.
</td>
</tr>
<tr>
<td>1719</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCher2k(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const float *beta, cuComplex *C, size_t ldc)</code><br>
Performs single-precision complex Hermitian rank-2k update.
</td>
</tr>
<tr>
<td>1720</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZher2k(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const double *beta, cuDoubleComplex *C, size_t ldc)</code><br>
Performs double-precision complex Hermitian rank-2k update.
</td>
</tr>
<tr>
<td>1721</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCherkx(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, const float *beta, cuComplex *C, size_t ldc)</code><br>
Performs single-precision complex Hermitian rank-kx update.
</td>
</tr>
<tr>
<td>1722</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZherkx(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, const double *beta, cuDoubleComplex *C, size_t ldc)</code><br>
Performs double-precision complex Hermitian rank-kx update.
</td>
</tr>
<tr>
<td>1723</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtStrsm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const float *alpha, const float *A, size_t lda, float *B, size_t ldb)</code><br>
Solves single-precision triangular linear system with multiple right-hand sides.
</td>
</tr>
<tr>
<td>1724</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDtrsm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const double *alpha, const double *A, size_t lda, double *B, size_t ldb)</code><br>
Solves double-precision triangular linear system with multiple right-hand sides.
</td>
</tr>
<tr>
<td>1725</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCtrsm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, cuComplex *B, size_t ldb)</code><br>
Solves single-precision complex triangular linear system with multiple right-hand sides.
</td>
</tr>
<tr>
<td>1726</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZtrsm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, cuDoubleComplex *B, size_t ldb)</code><br>
Solves double-precision complex triangular linear system with multiple right-hand sides.
</td>
</tr>
<tr>
<td>1727</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtStrmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const float *alpha, const float *A, size_t lda, const float *B, size_t ldb, float *C, size_t ldc)</code><br>
Performs single-precision triangular matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1728</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDtrmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const double *alpha, const double *A, size_t lda, const double *B, size_t ldb, double *C, size_t ldc)</code><br>
Performs double-precision triangular matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1729</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCtrmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const cuComplex *alpha, const cuComplex *A, size_t lda, const cuComplex *B, size_t ldb, cuComplex *C, size_t ldc)</code><br>
Performs single-precision complex triangular matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1730</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZtrmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, size_t lda, const cuDoubleComplex *B, size_t ldb, cuDoubleComplex *C, size_t ldc)</code><br>
Performs double-precision complex triangular matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1731</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSspmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const float *alpha, const float *AP, const float *B, size_t ldb, const float *beta, float *C, size_t ldc)</code><br>
Performs single-precision symmetric packed matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1732</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDspmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const double *alpha, const double *AP, const double *B, size_t ldb, const double *beta, double *C, size_t ldc)</code><br>
Performs double-precision symmetric packed matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1733</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCspmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *B, size_t ldb, const cuComplex *beta, cuComplex *C, size_t ldc)</code><br>
Performs single-precision complex symmetric packed matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1734</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtZspmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *B, size_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc)</code><br>
Performs double-precision complex symmetric packed matrix-matrix multiplication.
</td>
</tr>
<tr>
<td>1735</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
