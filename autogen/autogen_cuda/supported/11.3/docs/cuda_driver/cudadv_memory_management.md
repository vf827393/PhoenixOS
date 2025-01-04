<h2>PhOS Support: CUDA 11.3 - Driver APIs - Memory Management (0/67)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__MEM.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuArray3DCreate ( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray )</code><br>
Creates a 3D CUDA array.
</td>
</tr>
<tr>
<td>720</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuArray3DGetDescriptor ( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray )</code><br>
Get a 3D CUDA array descriptor.
</td>
</tr>
<tr>
<td>721</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuArrayCreate ( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray )</code><br>
Creates a 1D or 2D CUDA array.
</td>
</tr>
<tr>
<td>722</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuArrayDestroy ( CUarray hArray )</code><br>
Destroys a CUDA array.
</td>
</tr>
<tr>
<td>723</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuArrayGetDescriptor ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray )</code><br>
Get a 1D or 2D CUDA array descriptor.
</td>
</tr>
<tr>
<td>724</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuArrayGetPlane ( CUarray* pPlaneArray, CUarray hArray, unsigned int  planeIdx )</code><br>
Gets a CUDA array plane from a CUDA array.
</td>
</tr>
<tr>
<td>725</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuArrayGetSparseProperties ( CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array )</code><br>
Returns the layout properties of a sparse CUDA array.
</td>
</tr>
<tr>
<td>726</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId )</code><br>
Returns a handle to a compute device.
</td>
</tr>
<tr>
<td>727</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev )</code><br>
Returns a PCI Bus Id string for the device.
</td>
</tr>
<tr>
<td>728</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuIpcCloseMemHandle ( CUdeviceptr dptr )</code><br>
Attempts to close memory mapped with cuIpcOpenMemHandle.
</td>
</tr>
<tr>
<td>729</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuIpcGetEventHandle ( CUipcEventHandle* pHandle, CUevent event )</code><br>
Gets an interprocess handle for a previously allocated event.
</td>
</tr>
<tr>
<td>730</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr )</code><br>
Gets an interprocess memory handle for an existing device memory allocation.
</td>
</tr>
<tr>
<td>731</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuIpcOpenEventHandle ( CUevent* phEvent, CUipcEventHandle handle )</code><br>
Opens an interprocess event handle for use in the current process.
</td>
</tr>
<tr>
<td>732</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuIpcOpenMemHandle ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags )</code><br>
Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
</td>
</tr>
<tr>
<td>733</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAlloc ( CUdeviceptr* dptr, size_t bytesize )</code><br>
Allocates device memory.
</td>
</tr>
<tr>
<td>734</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAllocHost ( void** pp, size_t bytesize )</code><br>
Allocates page-locked host memory.
</td>
</tr>
<tr>
<td>735</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags )</code><br>
Allocates memory that will be automatically managed by the Unified Memory system.
</td>
</tr>
<tr>
<td>736</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAllocPitch ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes )</code><br>
Allocates pitched device memory.
</td>
</tr>
<tr>
<td>737</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemFree ( CUdeviceptr dptr )</code><br>
Frees device memory.
</td>
</tr>
<tr>
<td>738</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemFreeHost ( void* p )</code><br>
Frees page-locked host memory.
</td>
</tr>
<tr>
<td>739</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )</code><br>
Get information on memory allocations.
</td>
</tr>
<tr>
<td>740</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemGetInfo ( size_t* free, size_t* total )</code><br>
Gets free and total memory.
</td>
</tr>
<tr>
<td>741</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemHostAlloc ( void** pp, size_t bytesize, unsigned int  Flags )</code><br>
Allocates page-locked host memory.
</td>
</tr>
<tr>
<td>742</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemHostGetDevicePointer ( CUdeviceptr* pdptr, void* p, unsigned int  Flags )</code><br>
Passes back device pointer of mapped pinned memory.
</td>
</tr>
<tr>
<td>743</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemHostGetFlags ( unsigned int* pFlags, void* p )</code><br>
Passes back flags that were used for a pinned allocation.
</td>
</tr>
<tr>
<td>744</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemHostRegister ( void* p, size_t bytesize, unsigned int  Flags )</code><br>
Registers an existing host memory range for use by CUDA.
</td>
</tr>
<tr>
<td>745</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemHostUnregister ( void* p )</code><br>
Unregisters a memory range that was registered with cuMemHostRegister.
</td>
</tr>
<tr>
<td>746</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount )</code><br>
Copies memory.
</td>
</tr>
<tr>
<td>747</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy2D ( const CUDA_MEMCPY2D* pCopy )</code><br>
Copies memory for 2D arrays.
</td>
</tr>
<tr>
<td>748</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy2DAsync ( const CUDA_MEMCPY2D* pCopy, CUstream hStream )</code><br>
Copies memory for 2D arrays.
</td>
</tr>
<tr>
<td>749</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy2DUnaligned ( const CUDA_MEMCPY2D* pCopy )</code><br>
Copies memory for 2D arrays.
</td>
</tr>
<tr>
<td>750</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy3D ( const CUDA_MEMCPY3D* pCopy )</code><br>
Copies memory for 3D arrays.
</td>
</tr>
<tr>
<td>751</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy3DAsync ( const CUDA_MEMCPY3D* pCopy, CUstream hStream )</code><br>
Copies memory for 3D arrays.
</td>
</tr>
<tr>
<td>752</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy )</code><br>
Copies memory between contexts.
</td>
</tr>
<tr>
<td>753</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream )</code><br>
Copies memory between contexts asynchronously.
</td>
</tr>
<tr>
<td>754</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream )</code><br>
Copies memory asynchronously.
</td>
</tr>
<tr>
<td>755</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyAtoA ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount )</code><br>
Copies memory from Array to Array.
</td>
</tr>
<tr>
<td>756</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount )</code><br>
Copies memory from Array to Device.
</td>
</tr>
<tr>
<td>757</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyAtoH ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount )</code><br>
Copies memory from Array to Host.
</td>
</tr>
<tr>
<td>758</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyAtoHAsync ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream )</code><br>
Copies memory from Array to Host.
</td>
</tr>
<tr>
<td>759</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyDtoA ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount )</code><br>
Copies memory from Device to Array.
</td>
</tr>
<tr>
<td>760</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyDtoD ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount )</code><br>
Copies memory from Device to Device.
</td>
</tr>
<tr>
<td>761</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyDtoDAsync ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )</code><br>
Copies memory from Device to Device.
</td>
</tr>
<tr>
<td>762</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyDtoH ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount )</code><br>
Copies memory from Device to Host.
</td>
</tr>
<tr>
<td>763</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyDtoHAsync ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )</code><br>
Copies memory from Device to Host.
</td>
</tr>
<tr>
<td>764</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyHtoA ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount )</code><br>
Copies memory from Host to Array.
</td>
</tr>
<tr>
<td>765</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyHtoAAsync ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream )</code><br>
Copies memory from Host to Array.
</td>
</tr>
<tr>
<td>766</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount )</code><br>
Copies memory from Host to Device.
</td>
</tr>
<tr>
<td>767</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyHtoDAsync ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream )</code><br>
Copies memory from Host to Device.
</td>
</tr>
<tr>
<td>768</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount )</code><br>
Copies device memory between two contexts.
</td>
</tr>
<tr>
<td>769</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream )</code><br>
Copies device memory between two contexts asynchronously.
</td>
</tr>
<tr>
<td>770</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD16 ( CUdeviceptr dstDevice, unsigned short us, size_t N )</code><br>
Initializes device memory.
</td>
</tr>
<tr>
<td>771</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream )</code><br>
Sets device memory.
</td>
</tr>
<tr>
<td>772</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD2D16 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height )</code><br>
Initializes device memory.
</td>
</tr>
<tr>
<td>773</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream )</code><br>
Sets device memory.
</td>
</tr>
<tr>
<td>774</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD2D32 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height )</code><br>
Initializes device memory.
</td>
</tr>
<tr>
<td>775</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream )</code><br>
Sets device memory.
</td>
</tr>
<tr>
<td>776</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD2D8 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height )</code><br>
Initializes device memory.
</td>
</tr>
<tr>
<td>777</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream )</code><br>
Sets device memory.
</td>
</tr>
<tr>
<td>778</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD32 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N )</code><br>
Initializes device memory.
</td>
</tr>
<tr>
<td>779</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream )</code><br>
Sets device memory.
</td>
</tr>
<tr>
<td>780</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD8 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N )</code><br>
Initializes device memory.
</td>
</tr>
<tr>
<td>781</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream )</code><br>
Sets device memory.
</td>
</tr>
<tr>
<td>782</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels )</code><br>
Creates a CUDA mipmapped array.
</td>
</tr>
<tr>
<td>783</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray )</code><br>
Destroys a CUDA mipmapped array.
</td>
</tr>
<tr>
<td>784</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level )</code><br>
Gets a mipmap level of a CUDA mipmapped array.
</td>
</tr>
<tr>
<td>785</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMipmappedArrayGetSparseProperties ( CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap )</code><br>
Returns the layout properties of a sparse CUDA mipmapped array.
</td>
</tr>
<tr>
<td>786</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
