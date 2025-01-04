<h2>PhOS Support: CUDA 11.3 - cuBLAS Level-2 Functions (0/66)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#cublas-level-2-function-reference

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)</code><br>
Performs a matrix-vector operation using a band matrix (single precision).
</td>
</tr>
<tr>
<td>1200</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)</code><br>
Performs a matrix-vector operation using a band matrix (double precision).
</td>
</tr>
<tr>
<td>1201</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a band matrix (complex single precision).
</td>
</tr>
<tr>
<td>1202</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a band matrix (complex double precision).
</td>
</tr>
<tr>
<td>1203</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)</code><br>
Performs a matrix-vector operation using a general matrix (single precision).
</td>
</tr>
<tr>
<td>1204</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)</code><br>
Performs a matrix-vector operation using a general matrix (double precision).
</td>
</tr>
<tr>
<td>1205</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a general matrix (complex single precision).
</td>
</tr>
<tr>
<td>1206</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a general matrix (complex double precision).
</td>
</tr>
<tr>
<td>1207</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)</code><br>
Performs a rank-1 update of a general matrix (single precision).
</td>
</tr>
<tr>
<td>1208</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda)</code><br>
Performs a rank-1 update of a general matrix (double precision).
</td>
</tr>
<tr>
<td>1209</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)</code><br>
Performs a rank-1 update of a general matrix (complex single precision, unconjugated).
</td>
</tr>
<tr>
<td>1210</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)</code><br>
Performs a rank-1 update of a general matrix (complex single precision, conjugated).
</td>
</tr>
<tr>
<td>1211</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)</code><br>
Performs a rank-1 update of a general matrix (complex double precision, unconjugated).
</td>
</tr>
<tr>
<td>1212</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)</code><br>
Performs a rank-1 update of a general matrix (complex double precision, conjugated).
</td>
</tr>
<tr>
<td>1213</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)</code><br>
Performs a matrix-vector operation using a symmetric band matrix (single precision).
</td>
</tr>
<tr>
<td>1214</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)</code><br>
Performs a matrix-vector operation using a symmetric band matrix (double precision).
</td>
</tr>
<tr>
<td>1215</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *AP, const float *x, int incx, const float *beta, float *y, int incy)</code><br>
Performs a matrix-vector operation using a symmetric packed matrix (single precision).
</td>
</tr>
<tr>
<td>1216</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *AP, const double *x, int incx, const double *beta, double *y, int incy)</code><br>
Performs a matrix-vector operation using a symmetric packed matrix (double precision).
</td>
</tr>
<tr>
<td>1217</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP)</code><br>
Performs a rank-1 update of a symmetric packed matrix (single precision).
</td>
</tr>
<tr>
<td>1218</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP)</code><br>
Performs a rank-1 update of a symmetric packed matrix (double precision).
</td>
</tr>
<tr>
<td>1219</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *AP)</code><br>
Performs a rank-2 update of a symmetric packed matrix (single precision).
</td>
</tr>
<tr>
<td>1220</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *AP)</code><br>
Performs a rank-2 update of a symmetric packed matrix (double precision).
</td>
</tr>
<tr>
<td>1221</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)</code><br>
Performs a matrix-vector operation using a symmetric matrix (single precision).
</td>
</tr>
<tr>
<td>1222</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)</code><br>
Performs a matrix-vector operation using a symmetric matrix (double precision).
</td>
</tr>
<tr>
<td>1223</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda)</code><br>
Performs a rank-1 update of a symmetric matrix (single precision).
</td>
</tr>
<tr>
<td>1224</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *A, int lda)</code><br>
Performs a rank-1 update of a symmetric matrix (double precision).
</td>
</tr>
<tr>
<td>1225</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)</code><br>
Performs a rank-2 update of a symmetric matrix (single precision).
</td>
</tr>
<tr>
<td>1226</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda)</code><br>
Performs a rank-2 update of a symmetric matrix (double precision).
</td>
</tr>
<tr>
<td>1227</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation (single precision).
</td>
</tr>
<tr>
<td>1228</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation (double precision).
</td>
</tr>
<tr>
<td>1229</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation (complex single precision).
</td>
</tr>
<tr>
<td>1230</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation (complex double precision).
</td>
</tr>
<tr>
<td>1231</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation with a single right-hand side (single precision).
</td>
</tr>
<tr>
<td>1232</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation with a single right-hand side (double precision).
</td>
</tr>
<tr>
<td>1233</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation with a single right-hand side (complex single precision).
</td>
</tr>
<tr>
<td>1234</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)</code><br>
Solves a triangular banded matrix-vector equation with a single right-hand side (complex double precision).
</td>
</tr>
<tr>
<td>1235</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation (single precision).
</td>
</tr>
<tr>
<td>1236</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation (double precision).
</td>
</tr>
<tr>
<td>1237</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation (complex single precision).
</td>
</tr>
<tr>
<td>1238</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation (complex double precision).
</td>
</tr>
<tr>
<td>1239</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation with a single right-hand side (single precision).
</td>
</tr>
<tr>
<td>1240</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation with a single right-hand side (double precision).
</td>
</tr>
<tr>
<td>1241</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation with a single right-hand side (complex single precision).
</td>
</tr>
<tr>
<td>1242</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx)</code><br>
Solves a triangular packed matrix-vector equation with a single right-hand side (complex double precision).
</td>
</tr>
<tr>
<td>1243</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)</code><br>
Solves a triangular matrix-vector equation (single precision).
</td>
</tr>
<tr>
<td>1244</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)</code><br>
Solves a triangular matrix-vector equation (double precision).
</td>
</tr>
<tr>
<td>1245</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx)</code><br>
Solves a triangular matrix-vector equation (complex single precision).
</td>
</tr>
<tr>
<td>1246</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)</code><br>
Solves a triangular matrix-vector equation (complex double precision).
</td>
</tr>
<tr>
<td>1247</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)</code><br>
Solves a triangular matrix-vector equation with a single right-hand side (single precision).
</td>
</tr>
<tr>
<td>1248</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx)</code><br>
Solves a triangular matrix-vector equation with a single right-hand side (double precision).
</td>
</tr>
<tr>
<td>1249</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx)</code><br>
Solves a triangular matrix-vector equation with a single right-hand side (complex single precision).
</td>
</tr>
<tr>
<td>1250</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx)</code><br>
Solves a triangular matrix-vector equation with a single right-hand side (complex double precision).
</td>
</tr>
<tr>
<td>1251</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a Hermitian matrix (complex single precision).
</td>
</tr>
<tr>
<td>1252</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a Hermitian matrix (complex double precision).
</td>
</tr>
<tr>
<td>1253</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a Hermitian band matrix (complex single precision).
</td>
</tr>
<tr>
<td>1254</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a Hermitian band matrix (complex double precision).
</td>
</tr>
<tr>
<td>1255</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a Hermitian packed matrix (complex single precision).
</td>
</tr>
<tr>
<td>1256</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy)</code><br>
Performs a matrix-vector operation using a Hermitian packed matrix (complex double precision).
</td>
</tr>
<tr>
<td>1257</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *A, int lda)</code><br>
Performs a rank-1 update of a Hermitian matrix (complex single precision, real scalar).
</td>
</tr>
<tr>
<td>1258</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda)</code><br>
Performs a rank-1 update of a Hermitian matrix (complex double precision, real scalar).
</td>
</tr>
<tr>
<td>1259</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda)</code><br>
Performs a rank-2 update of a Hermitian matrix (complex single precision).
</td>
</tr>
<tr>
<td>1260</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda)</code><br>
Performs a rank-2 update of a Hermitian matrix (complex double precision).
</td>
</tr>
<tr>
<td>1261</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP)</code><br>
Performs a rank-1 update of a Hermitian packed matrix (complex single precision, real scalar).
</td>
</tr>
<tr>
<td>1262</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP)</code><br>
Performs a rank-1 update of a Hermitian packed matrix (complex double precision, real scalar).
</td>
</tr>
<tr>
<td>1263</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP)</code><br>
Performs a rank-2 update of a Hermitian packed matrix (complex single precision).
</td>
</tr>
<tr>
<td>1264</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP)</code><br>
Performs a rank-2 update of a Hermitian packed matrix (complex double precision).
</td>
</tr>
<tr>
<td>1265</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
