#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

namespace cosmico {

void snapshotReadbackCuda(void* hostDst, const void* devicePtr, size_t bytes) {
    cudaError_t err = cudaMemcpy(hostDst, devicePtr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        fprintf(stderr, "[SnapshotCuda] readback failed: %s\n", cudaGetErrorString(err));
}

void snapshotUploadCuda(void* devicePtr, const void* hostSrc, size_t bytes) {
    cudaError_t err = cudaMemcpy(devicePtr, hostSrc, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        fprintf(stderr, "[SnapshotCuda] upload failed: %s\n", cudaGetErrorString(err));
}

} // namespace cosmico
