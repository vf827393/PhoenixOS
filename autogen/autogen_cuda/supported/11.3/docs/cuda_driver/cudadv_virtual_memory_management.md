<h2>PhOS Support: CUDA 11.3 - Driver APIs - Virtual Memory Management (0/14)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__MEMORY.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAddressFree ( CUdeviceptr ptr, size_t size )</code><br>
Free an address range reservation.
</td>
</tr>
<tr>
<td>800</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAddressReserve ( CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags )</code><br>
Allocate an address range reservation.
</td>
</tr>
<tr>
<td>801</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemCreate ( CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags )</code><br>
Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.
</td>
</tr>
<tr>
<td>802</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemExportToShareableHandle ( void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags )</code><br>
Exports an allocation to a requested shareable handle type.
</td>
</tr>
<tr>
<td>803</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemGetAccess ( unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr )</code><br>
Get the access flags set for the given location and ptr.
</td>
</tr>
<tr>
<td>804</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemGetAllocationGranularity ( size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option )</code><br>
Calculates either the minimal or recommended granularity.
</td>
</tr>
<tr>
<td>805</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemGetAllocationPropertiesFromHandle ( CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle )</code><br>
Retrieve the contents of the property structure defining properties for this handle.
</td>
</tr>
<tr>
<td>806</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemImportFromShareableHandle ( CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType )</code><br>
Imports an allocation from a requested shareable handle type.
</td>
</tr>
<tr>
<td>807</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemMap ( CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags )</code><br>
Maps an allocation handle to a reserved virtual address range.
</td>
</tr>
<tr>
<td>808</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemMapArrayAsync ( CUarrayMapInfo* mapInfoList, unsigned int  count, CUstream hStream )</code><br>
Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
</td>
</tr>
<tr>
<td>809</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemRelease ( CUmemGenericAllocationHandle handle )</code><br>
Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.
</td>
</tr>
<tr>
<td>810</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemRetainAllocationHandle ( CUmemGenericAllocationHandle* handle, void* addr )</code><br>
Given an address addr, returns the allocation handle of the backing memory allocation.
</td>
</tr>
<tr>
<td>811</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemSetAccess ( CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count )</code><br>
Set the access flags for each location specified in desc for the given virtual address range.
</td>
</tr>
<tr>
<td>812</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemUnmap ( CUdeviceptr ptr, size_t size )</code><br>
Unmap the backing memory of a given address range.
</td>
</tr>
<tr>
<td>813</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
</p>
