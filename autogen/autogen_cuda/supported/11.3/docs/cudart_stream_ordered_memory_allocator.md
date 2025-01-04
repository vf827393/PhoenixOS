<h2>PhOS Support: CUDA 11.3 - Runtime APIs - Stream Ordered Memory Allocator (0/14)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaFreeAsync ( void* devPtr, cudaStream_t hStream )</code><br>
Frees memory with stream ordered semantics.
</td>
</td>
</tr>
<tr>
<td>380</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMallocAsync ( void** devPtr, size_t size, cudaStream_t hStream )</code><br>
Allocates memory with stream ordered semantics.
</td>
</tr>
<tr>
<td>381</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMallocFromPoolAsync ( void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream )</code><br>
Allocates memory from a specified pool with stream ordered semantics.
</td>
</tr>
<tr>
<td>382</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolCreate ( cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps )</code><br>
Creates a memory pool.
</td>
</tr>
<tr>
<td>383</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolDestroy ( cudaMemPool_t memPool )</code><br>
Destroys the specified memory pool.
</td>
</tr>
<tr>
<td>384</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolExportPointer ( cudaMemPoolPtrExportData* exportData, void* ptr )</code><br>
Export data to share a memory pool allocation between processes.
</td>
</tr>
<tr>
<td>385</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolExportToShareableHandle ( void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int  flags )</code><br>
Exports a memory pool to the requested handle type.
</td>
</tr>
<tr>
<td>386</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolGetAccess ( cudaMemAccessFlags ** flags, cudaMemPool_t memPool, cudaMemLocation* location )</code><br>
Returns the accessibility of a pool from a device.
</td>
</tr>
<tr>
<td>387</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolGetAttribute ( cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value )</code><br>
Gets attributes of a memory pool.
</td>
</tr>
<tr>
<td>388</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolImportFromShareableHandle ( cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int  flags )</code><br>
imports a memory pool from a shared handle.
</td>
</tr>
<tr>
<td>389</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolImportPointer ( void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData )</code><br>
Import a memory pool allocation from another process.
</td>
</tr>
<tr>
<td>390</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolSetAccess ( cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count )</code><br>
Controls visibility of pools between devices.
</td>
</tr>
<tr>
<td>391</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolSetAttribute ( cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value )</code><br>
Sets attributes of a memory pool.
</td>
</tr>
<tr>
<td>392</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaMemPoolTrimTo ( cudaMemPool_t memPool, size_t minBytesToKeep )</code><br>
Tries to release memory back to the OS.
</td>
</tr>
<tr>
<td>393</td>
<td>✗</td>
<td>✗</td>
</tr>
</table> 
