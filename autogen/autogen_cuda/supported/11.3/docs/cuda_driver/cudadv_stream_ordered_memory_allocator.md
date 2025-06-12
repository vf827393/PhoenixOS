<h2>PhOS Support: CUDA 11.3 - Driver APIs - Stream Ordered Memory Allocator (0/14)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__MEMORY__POOLS.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAllocAsync ( CUdeviceptr* dptr, size_t bytesize, CUstream hStream )</code><br>
Allocates memory with stream ordered semantics.
</td>
</tr>
<tr>
<td>820</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAllocFromPoolAsync ( CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream )</code><br>
Allocates memory from a specified pool with stream ordered semantics.
</td>
</tr>
<tr>
<td>821</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemFreeAsync ( CUdeviceptr dptr, CUstream hStream )</code><br>
Frees memory with stream ordered semantics.
</td>
</tr>
<tr>
<td>822</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolCreate ( CUmemoryPool* pool, const CUmemPoolProps* poolProps )</code><br>
Creates a memory pool.
</td>
</tr>
<tr>
<td>823</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolDestroy ( CUmemoryPool pool )</code><br>
Destroys the specified memory pool.
</td>
</tr>
<tr>
<td>824</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolExportPointer ( CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr )</code><br>
Export data to share a memory pool allocation between processes.
</td>
</tr>
<tr>
<td>825</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolExportToShareableHandle ( void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags )</code><br>
Exports a memory pool to the requested handle type.
</td>
</tr>
<tr>
<td>826</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolGetAccess ( CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location )</code><br>
Returns the accessibility of a pool from a device.
</td>
</tr>
<tr>
<td>827</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolGetAttribute ( CUmemoryPool pool, CUmemPool_attribute attr, void* value )</code><br>
Gets attributes of a memory pool.
</td>
</tr>
<tr>
<td>828</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolImportFromShareableHandle ( CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags )</code><br>
Imports a memory pool from a shared handle.
</td>
</tr>
<tr>
<td>829</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolImportPointer ( CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData )</code><br>
Import a memory pool allocation from another process.
</td>
</tr>
<tr>
<td>830</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolSetAccess ( CUmemoryPool pool, const CUmemAccessDesc* map, size_t count )</code><br>
Controls visibility of pools between devices.
</td>
</tr>
<tr>
<td>831</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolSetAttribute ( CUmemoryPool pool, CUmemPool_attribute attr, void* value )</code><br>
Sets attributes of a memory pool.
</td>
</tr>
<tr>
<td>832</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPoolTrimTo ( CUmemoryPool pool, size_t minBytesToKeep )</code><br>
Tries to release memory back to the OS.
</td>
</tr>
<tr>
<td>833</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
