<h2>PhOS Support: CUDA 11.3 - Runtime APIs - Memory Management (0/57)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaArrayGetInfo ( cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array )</code><br>
Gets info about the specified cudaArray.
</td>
</tr>
<tr>
<td>250</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaArrayGetPlane ( cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int  planeIdx )</code><br>
Gets a CUDA array plane from a CUDA array.
</td>
</tr>
<tr>
<td>251</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaArrayGetSparseProperties ( cudaArraySparseProperties* sparseProperties, cudaArray_t array )</code><br>
Returns the layout properties of a sparse CUDA array.
</td>
</tr>
<tr>
<td>252</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t  cudaFree ( void* devPtr )</code><br>
Frees memory on the device.
</td>
</tr>
<tr>
<td>253</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaFreeArray ( cudaArray_t array )</code><br>
Frees an array on the device.
</td>
</tr>
<tr>
<td>254</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaFreeHost ( void* ptr )</code><br>
Frees page-locked memory.
</td>
</tr>
<tr>
<td>255</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaFreeMipmappedArray ( cudaMipmappedArray_t mipmappedArray )</code><br>
Frees a mipmapped array on the device.
</td>
</tr>
<tr>
<td>256</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetMipmappedArrayLevel ( cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level )</code><br>
Gets a mipmap level of a CUDA mipmapped array.
</td>
</tr>
<tr>
<td>257</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetSymbolAddress ( void** devPtr, const void* symbol )</code><br>
Finds the address associated with a CUDA symbol.
</td>
</tr>
<tr>
<td>258</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetSymbolSize ( size_t* size, const void* symbol )</code><br>
Finds the size of the object associated with a CUDA symbol.
</td>
</tr>
<tr>
<td>259</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags )</code><br>
Allocates page-locked memory on the host.
</td>
</tr>
<tr>
<td>260</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaHostGetDevicePointer ( void** pDevice, void* pHost, unsigned int  flags )</code><br>
Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
</td>
</tr>
<tr>
<td>261</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaHostGetFlags ( unsigned int* pFlags, void* pHost )</code><br>
Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.
</td>
</tr>
<tr>
<td>262</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags )</code><br>
Registers an existing host memory range for use by CUDA.
</td>
</tr>
<tr>
<td>263</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaHostUnregister ( void* ptr )</code><br>
Unregisters a memory range that was registered with cudaHostRegister.
</td>
</tr>
<tr>
<td>264</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMalloc ( void** devPtr, size_t size )</code><br>
Allocate memory on the device.
</td>
</tr>
<tr>
<td>265</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMalloc3D ( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent )</code><br>
Allocates logical 1D, 2D, or 3D memory objects on the device.
</td>
</tr>
<tr>
<td>266</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMalloc3DArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  flags = 0 )</code><br>
Allocate an array on the device.
</td>
</tr>
<tr>
<td>267</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMallocArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height = 0, unsigned int  flags = 0 )</code><br>
Allocate an array on the device.
</td>
</tr>
<tr>
<td>268</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMallocHost ( void** ptr, size_t size )</code><br>
Allocates page-locked memory on the host.
</td>
</tr>
<tr>
<td>269</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal )</code><br>
Allocates memory that will be automatically managed by the Unified Memory system.
</td>
</tr>
<tr>
<td>270</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMallocMipmappedArray ( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags = 0 )</code><br>
Allocate a mipmapped array on the device.
</td>
</tr>
<tr>
<td>271</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height )</code><br>
Allocates pitched memory on the device.
</td>
</tr>
<tr>
<td>272</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemAdvise ( const void* devPtr, size_t count, cudaMemoryAdvise advice, int  device )</code><br>
Advise about the usage of a given memory range.
</td>
</tr>
<tr>
<td>273</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemGetInfo ( size_t* free, size_t* total )</code><br>
Gets free and total device memory.
</td>
</tr>
<tr>
<td>274</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPrefetchAsync ( const void* devPtr, size_t count, int  dstDevice, cudaStream_t stream = 0 )</code><br>
Prefetches memory to the specified destination device.
</td>
</tr>
<tr>
<td>275</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemRangeGetAttribute ( void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count )</code><br>
Query an attribute of a given memory range.
</td>
</tr>
<tr>
<td>276</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemRangeGetAttributes ( void** data, size_t* dataSizes, cudaMemRangeAttribute ** attributes, size_t numAttributes, const void* devPtr, size_t count )</code><br>
Query attributes of a given memory range.
</td>
</tr>
<tr>
<td>277</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>278</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>279</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>280</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy2DAsync ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>281</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy2DFromArray ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>282</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>283</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy2DToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>284</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>285</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy3D ( const cudaMemcpy3DParms* p )</code><br>
Copies data between 3D objects.
</td>
</tr>
<tr>
<td>286</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p, cudaStream_t stream = 0 )</code><br>
Copies data between 3D objects.
</td>
</tr>
<tr>
<td>287</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p )</code><br>
Copies memory between devices.
</td>
</tr>
<tr>
<td>288</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p, cudaStream_t stream = 0 )</code><br>
Copies memory between devices asynchronously.
</td>
</tr>
<tr>
<td>289</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device.
</td>
</tr>
<tr>
<td>290</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyFromSymbol ( void* dst, const void* symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost )</code><br>
Copies data from the given symbol on the device.
</td>
</tr>
<tr>
<td>291</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyFromSymbolAsync ( void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data from the given symbol on the device.
</td>
</tr>
<tr>
<td>292</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyPeer ( void* dst, int dstDevice, const void* src, int srcDevice, size_t count )</code><br>
Copies memory between two devices.
</td>
</tr>
<tr>
<td>293</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyPeerAsync ( void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream = 0 )</code><br>
Copies memory between two devices asynchronously.
</td>
</tr>
<tr>
<td>294</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice )</code><br>
Copies data to the given symbol on the device.
</td>
</tr>
<tr>
<td>295</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyToSymbolAsync ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data to the given symbol on the device.
</td>
</tr>
<tr>
<td>296</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemset ( void* devPtr, int value, size_t count )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>297</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemset2D ( void* devPtr, size_t pitch, int value, size_t width, size_t height )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>298</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemset2DAsync ( void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0 )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>299</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemset3D ( cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>300</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemset3DAsync ( cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0 )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>301</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemsetAsync ( void* devPtr, int value, size_t count, cudaStream_t stream = 0 )</code><br>
Initializes or sets device memory to a value.
</td>
</tr>
<tr>
<td>302</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMipmappedArrayGetSparseProperties ( cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap )</code><br>
Returns the layout properties of a sparse CUDA mipmapped array.
</td>
</tr>
<tr>
<td>303</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaExtent make_cudaExtent ( size_t w, size_t h, size_t d )</code><br>
Returns a cudaExtent based on input parameters.
</td>
</tr>
<tr>
<td>304</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaPitchedPtr make_cudaPitchedPtr ( void* d, size_t p, size_t xsz, size_t ysz )</code><br>
Returns a cudaPitchedPtr based on input parameters.
</td>
</tr>
<tr>
<td>305</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaPos make_cudaPos ( size_t x, size_t y, size_t z )</code><br>
Returns a cudaPos based on input parameters.
</td>
</tr>
<tr>
<td>306</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyH2D ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Copies data between host and device (Host to device)
</td>
</tr>
<tr>
<td>320</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyD2H ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Copies data between host and device (Device to host)
</td>
</tr>
<tr>
<td>321</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemcpyD2D ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Copies data between host and device (Device to device)
</td>
</tr>
<tr>
<td>322</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>​cudaError_t cudaMemcpyAsyncH2D ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device (Host to device)
</td>
</tr>
<tr>
<td>323</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>​cudaError_t cudaMemcpyAsyncD2H ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device (Device to host)
</td>
</tr>
<tr>
<td>324</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>​cudaError_t cudaMemcpyAsyncD2D ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )</code><br>
Copies data between host and device (Device to device)
</td>
</tr>
<tr>
<td>325</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
