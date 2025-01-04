<h2>PhOS Support: CUDA 11.3 - cuBLAS Level-3 Functions (0/80)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#cublas-level-3-function-reference

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)</code><br>
Performs matrix-matrix multiplication (single precision).
</td>
</tr>
<tr>
<td>1300</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)</code><br>
Performs matrix-matrix multiplication (double precision).
</td>
</tr>
<tr>
<td>1301</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs matrix-matrix multiplication (complex single precision).
</td>
</tr>
<tr>
<td>1302</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs matrix-matrix multiplication (complex double precision).
</td>
</tr>
<tr>
<td>1303</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc)</code><br>
Performs matrix-matrix multiplication (half precision).
</td>
</tr>
<tr>
<td>1304</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)</code><br>
Performs matrix-matrix multiplication using 3m algorithm (single precision).
</td>
</tr>
<tr>
<td>1305</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)</code><br>
Performs matrix-matrix multiplication using 3m algorithm (double precision).
</td>
</tr>
<tr>
<td>1306</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs matrix-matrix multiplication using 3m algorithm (complex single precision).
</td>
</tr>
<tr>
<td>1307</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs matrix-matrix multiplication using 3m algorithm (complex double precision).
</td>
</tr>
<tr>
<td>1308</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *Aarray[], int lda, const float *Barray[], int ldb, const float *beta, float *Carray[], int ldc, int batchCount)</code><br>
Performs batched matrix-matrix multiplication (single precision).
</td>
</tr>
<tr>
<td>1309</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *Aarray[], int lda, const double *Barray[], int ldb, const double *beta, double *Carray[], int ldc, int batchCount)</code><br>
Performs batched matrix-matrix multiplication (double precision).
</td>
</tr>
<tr>
<td>1310</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *Aarray[], int lda, const cuComplex *Barray[], int ldb, const cuComplex *beta, cuComplex *Carray[], int ldc, int batchCount)</code><br>
Performs batched matrix-matrix multiplication (complex single precision).
</td>
</tr>
<tr>
<td>1311</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *Aarray[], int lda, const cuDoubleComplex *Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex *Carray[], int ldc, int batchCount)</code><br>
Performs batched matrix-matrix multiplication (complex double precision).
</td>
</tr>
<tr>
<td>1312</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount)</code><br>
Performs strided batched matrix-matrix multiplication (single precision).
</td>
</tr>
<tr>
<td>1313</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount)</code><br>
Performs strided batched matrix-matrix multiplication (double precision).
</td>
</tr>
<tr>
<td>1314</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount)</code><br>
Performs strided batched matrix-matrix multiplication (complex single precision).
</td>
</tr>
<tr>
<td>1315</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount)</code><br>
Performs strided batched matrix-matrix multiplication (complex double precision).
</td>
</tr>
<tr>
<td>1316</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)</code><br>
Performs symmetric matrix-matrix multiplication (single precision).
</td>
</tr>
<tr>
<td>1317</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)</code><br>
Performs symmetric matrix-matrix multiplication (double precision).
</td>
</tr>
<tr>
<td>1318</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs symmetric matrix-matrix multiplication (complex single precision).
</td>
</tr>
<tr>
<td>1319</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs symmetric matrix-matrix multiplication (complex double precision).
</td>
</tr>
<tr>
<td>1320</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc)</code><br>
Performs symmetric rank-k update (single precision).
</td>
</tr>
<tr>
<td>1321</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc)</code><br>
Performs symmetric rank-k update (double precision).
</td>
</tr>
<tr>
<td>1322</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs symmetric rank-k update (complex single precision).
</td>
</tr>
<tr>
<td>1323</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs symmetric rank-k update (complex double precision).
</td>
</tr>
<tr>
<td>1324</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)</code><br>
Performs symmetric rank-2k update (single precision).
</td>
</tr>
<tr>
<td>1325</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)</code><br>
Performs symmetric rank-2k update (double precision).
</td>
</tr>
<tr>
<td>1326</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs symmetric rank-2k update (complex single precision).
</td>
</tr>
<tr>
<td>1327</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs symmetric rank-2k update (complex double precision).
</td>
</tr>
<tr>
<td>1328</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)</code><br>
Performs symmetric rank-k update with an extra matrix (single precision).
</td>
</tr>
<tr>
<td>1329</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)</code><br>
Performs symmetric rank-k update with an extra matrix (double precision).
</td>
</tr>
<tr>
<td>1330</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs symmetric rank-k update with an extra matrix (complex single precision).
</td>
</tr>
<tr>
<td>1331</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs symmetric rank-k update with an extra matrix (complex double precision).
</td>
</tr>
<tr>
<td>1332</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb)</code><br>
Performs triangular matrix-matrix multiplication (single precision).
</td>
</tr>
<tr>
<td>1333</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb)</code><br>
Performs triangular matrix-matrix multiplication (double precision).
</td>
</tr>
<tr>
<td>1334</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb)</code><br>
Performs triangular matrix-matrix multiplication (complex single precision).
</td>
</tr>
<tr>
<td>1335</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb)</code><br>
Performs triangular matrix-matrix multiplication (complex double precision).
</td>
</tr>
<tr>
<td>1336</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb)</code><br>
Solves triangular matrix equations (single precision).
</td>
</tr>
<tr>
<td>1337</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb)</code><br>
Solves triangular matrix equations (double precision).
</td>
</tr>
<tr>
<td>1338</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb)</code><br>
Solves triangular matrix equations (complex single precision).
</td>
</tr>
<tr>
<td>1339</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb)</code><br>
Solves triangular matrix equations (complex double precision).
</td>
</tr>
<tr>
<td>1340</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *Aarray[], int lda, float *Barray[], int ldb, int batchCount)</code><br>
Solves batched triangular matrix equations (single precision).
</td>
</tr>
<tr>
<td>1341</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *Aarray[], int lda, double *Barray[], int ldb, int batchCount)</code><br>
Solves batched triangular matrix equations (double precision).
</td>
</tr>
<tr>
<td>1342</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *Aarray[], int lda, cuComplex *Barray[], int ldb, int batchCount)</code><br>
Solves batched triangular matrix equations (complex single precision).
</td>
</tr>
<tr>
<td>1343</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *Aarray[], int lda, cuDoubleComplex *Barray[], int ldb, int batchCount)</code><br>
Solves batched triangular matrix equations (complex double precision).
</td>
</tr>
<tr>
<td>1344</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs Hermitian matrix-matrix multiplication (complex single precision).
</td>
</tr>
<tr>
<td>1345</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs Hermitian matrix-matrix multiplication (complex double precision).
</td>
</tr>
<tr>
<td>1346</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc)</code><br>
Performs Hermitian rank-k update (complex single precision, real scalar).
</td>
</tr>
<tr>
<td>1347</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs Hermitian rank-k update (complex double precision, real scalar).
</td>
</tr>
<tr>
<td>1348</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc)</code><br>
Performs Hermitian rank-2k update (complex single precision, real scalar).
</td>
</tr>
<tr>
<td>1349</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs Hermitian rank-2k update (complex double precision, real scalar).
</td>
</tr>
<tr>
<td>1350</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc)</code><br>
Performs Hermitian rank-k update with an extra matrix (complex single precision).
</td>
</tr>
<tr>
<td>1351</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)</code><br>
Performs Hermitian rank-k update with an extra matrix (complex double precision).
</td>
</tr>
<tr>
<td>1352</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
