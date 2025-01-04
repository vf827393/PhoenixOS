<h2>PhOS Support: CUDA 11.3 - cuBLAS Level-1 Functions (0/52)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#cublas-level-1-function-reference

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x, int incx, int *result)</code><br>
Finds the index of the element with the maximum absolute value (single precision).
</td>
</tr>
<tr>
<td>1140</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double *x, int incx, int *result)</code><br>
Finds the index of the element with the maximum absolute value (double precision).
</td>
</tr>
<tr>
<td>1141</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result)</code><br>
Finds the index of the element with the maximum absolute value (complex single precision).
</td>
</tr>
<tr>
<td>1142</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result)</code><br>
Finds the index of the element with the maximum absolute value (complex double precision).
</td>
</tr>
<tr>
<td>1143</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float *x, int incx, int *result)</code><br>
Finds the index of the element with the minimum absolute value (single precision).
</td>
</tr>
<tr>
<td>1144</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double *x, int incx, int *result)</code><br>
Finds the index of the element with the minimum absolute value (double precision).
</td>
</tr>
<tr>
<td>1145</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result)</code><br>
Finds the index of the element with the minimum absolute value (complex single precision).
</td>
</tr>
<tr>
<td>1146</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result)</code><br>
Finds the index of the element with the minimum absolute value (complex double precision).
</td>
</tr>
<tr>
<td>1147</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float *x, int incx, float *result)</code><br>
Computes the sum of the absolute values of the elements (single precision).
</td>
</tr>
<tr>
<td>1148</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double *x, int incx, double *result)</code><br>
Computes the sum of the absolute values of the elements (double precision).
</td>
</tr>
<tr>
<td>1149</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasScasum(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result)</code><br>
Computes the sum of the absolute values of the elements (complex single precision).
</td>
</tr>
<tr>
<td>1150</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDzasum(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result)</code><br>
Computes the sum of the absolute values of the elements (complex double precision).
</td>
</tr>
<tr>
<td>1151</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)</code><br>
Computes the sum of a vector scaled by a scalar and another vector (single precision).
</td>
</tr>
<tr>
<td>1152</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)</code><br>
Computes the sum of a vector scaled by a scalar and another vector (double precision).
</td>
</tr>
<tr>
<td>1153</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy)</code><br>
Computes the sum of a vector scaled by a scalar and another vector (complex single precision).
</td>
</tr>
<tr>
<td>1154</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)</code><br>
Computes the sum of a vector scaled by a scalar and another vector (complex double precision).
</td>
</tr>
<tr>
<td>1155</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasScopy(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy)</code><br>
Copies a vector to another vector (single precision).
</td>
</tr>
<tr>
<td>1156</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDcopy(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy)</code><br>
Copies a vector to another vector (double precision).
</td>
</tr>
<tr>
<td>1157</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCcopy(cublasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy)</code><br>
Copies a vector to another vector (complex single precision).
</td>
</tr>
<tr>
<td>1158</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZcopy(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)</code><br>
Copies a vector to another vector (complex double precision).
</td>
</tr>
<tr>
<td>1159</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result)</code><br>
Computes the dot product of two vectors (single precision).
</td>
</tr>
<tr>
<td>1160</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDdot(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result)</code><br>
Computes the dot product of two vectors (double precision).
</td>
</tr>
<tr>
<td>1161</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)</code><br>
Computes the dot product of two vectors (complex single precision, unconjugated).
</td>
</tr>
<tr>
<td>1162</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)</code><br>
Computes the dot product of two vectors (complex single precision, conjugated).
</td>
</tr>
<tr>
<td>1163</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZdotu(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)</code><br>
Computes the dot product of two vectors (complex double precision, unconjugated).
</td>
</tr>
<tr>
<td>1164</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZdotc(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)</code><br>
Computes the dot product of two vectors (complex double precision, conjugated).
</td>
</tr>
<tr>
<td>1165</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float *x, int incx, float *result)</code><br>
Computes the Euclidean norm of a vector (single precision).
</td>
</tr>
<tr>
<td>1166</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double *x, int incx, double *result)</code><br>
Computes the Euclidean norm of a vector (double precision).
</td>
</tr>
<tr>
<td>1167</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result)</code><br>
Computes the Euclidean norm of a vector (complex single precision).
</td>
</tr>
<tr>
<td>1168</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result)</code><br>
Computes the Euclidean norm of a vector (complex double precision).
</td>
</tr>
<tr>
<td>1169</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSrot(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s)</code><br>
Applies a Givens rotation to a vector (single precision).
</td>
</tr>
<tr>
<td>1170</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDrot(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s)</code><br>
Applies a Givens rotation to a vector (double precision).
</td>
</tr>
<tr>
<td>1171</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCrot(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s)</code><br>
Applies a Givens rotation to a vector (complex single precision).
</td>
</tr>
<tr>
<td>1172</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s)</code><br>
Applies a Givens rotation to a vector (complex double precision).
</td>
</tr>
<tr>
<td>1173</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSrotg(cublasHandle_t handle, float *a, float *b, float *c, float *s)</code><br>
Constructs a Givens rotation (single precision).
</td>
</tr>
<tr>
<td>1174</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDrotg(cublasHandle_t handle, double *a, double *b, double *c, double *s)</code><br>
Constructs a Givens rotation (double precision).
</td>
</tr>
<tr>
<td>1175</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCrotg(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s)</code><br>
Constructs a Givens rotation (complex single precision).
</td>
</tr>
<tr>
<td>1176</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZrotg(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s)</code><br>
Constructs a Givens rotation (complex double precision).
</td>
</tr>
<tr>
<td>1177</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param)</code><br>
Applies a modified Givens rotation to a vector (single precision).
</td>
</tr>
<tr>
<td>1178</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *param)</code><br>
Applies a modified Givens rotation to a vector (double precision).
</td>
</tr>
<tr>
<td>1179</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSrotmg(cublasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param)</code><br>
Constructs a modified Givens rotation (single precision).
</td>
</tr>
<tr>
<td>1180</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDrotmg(cublasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double *param)</code><br>
Constructs a modified Givens rotation (double precision).
</td>
</tr>
<tr>
<td>1181</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx)</code><br>
Scales a vector by a scalar (single precision).
</td>
</tr>
<tr>
<td>1182</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx)</code><br>
Scales a vector by a scalar (double precision).
</td>
</tr>
<tr>
<td>1183</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCscal(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx)</code><br>
Scales a vector by a scalar (complex single precision).
</td>
</tr>
<tr>
<td>1184</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZscal(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx)</code><br>
Scales a vector by a scalar (complex double precision).
</td>
</tr>
<tr>
<td>1185</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx)</code><br>
Scales a vector by a scalar (complex single precision, real scalar).
</td>
</tr>
<tr>
<td>1186</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx)</code><br>
Scales a vector by a scalar (complex double precision, real scalar).
</td>
</tr>
<tr>
<td>1187</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy)</code><br>
Swaps the elements of two vectors (single precision).
</td>
</tr>
<tr>
<td>1188</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy)</code><br>
Swaps the elements of two vectors (double precision).
</td>
</tr>
<tr>
<td>1189</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy)</code><br>
Swaps the elements of two vectors (complex single precision).
</td>
</tr>
<tr>
<td>1190</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)</code><br>
Swaps the elements of two vectors (complex double precision).
</td>
</tr>
<tr>
<td>1191</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
