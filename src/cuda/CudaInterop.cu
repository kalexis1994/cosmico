#include <cosmico/cuda/CudaInterop.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA] Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

namespace cosmico {

void CudaInterop::init() {
    if (m_initialized) return;

    // Initialize CUDA
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA-capable device found");
    }

    // Use device 0 (same GPU as Vulkan)
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    fprintf(stderr, "[CUDA] Device: %s (SM %d.%d, %zu MB)\n",
            props.name, props.major, props.minor,
            props.totalGlobalMem / (1024 * 1024));

    // Create a CUDA stream for kernel execution
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    m_stream = stream;

    m_initialized = true;
    fprintf(stderr, "[CUDA] Interop initialized\n");
}

void CudaInterop::destroy() {
    if (!m_initialized) return;

    // Destroy imported semaphore
    if (m_extSemaphore) {
        cudaDestroyExternalSemaphore(static_cast<cudaExternalSemaphore_t>(m_extSemaphore));
        m_extSemaphore = nullptr;
    }

    // Free imported buffers
    for (int i = 0; i < m_importedCount; i++) {
        if (m_importedBuffers[i].devicePtr) {
            cudaFree(m_importedBuffers[i].devicePtr);
        }
        if (m_importedBuffers[i].extMemory) {
            cudaDestroyExternalMemory(static_cast<cudaExternalMemory_t>(m_importedBuffers[i].extMemory));
        }
    }
    m_importedCount = 0;

    // Destroy stream
    if (m_stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(m_stream));
        m_stream = nullptr;
    }

    m_initialized = false;
}

void* CudaInterop::importBuffer(HANDLE memoryHandle, size_t size) {
    if (m_importedCount >= MAX_IMPORTED_BUFFERS) {
        throw std::runtime_error("Too many imported buffers");
    }

    // Import external memory from Vulkan via Win32 handle
    cudaExternalMemoryHandleDesc extMemDesc{};
    extMemDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    extMemDesc.handle.win32.handle = memoryHandle;
    extMemDesc.size = size;
    extMemDesc.flags = cudaExternalMemoryDedicated;

    cudaExternalMemory_t extMem;
    CUDA_CHECK(cudaImportExternalMemory(&extMem, &extMemDesc));

    // Map to a CUDA device pointer
    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size = size;
    bufDesc.flags = 0;

    void* devicePtr = nullptr;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devicePtr, extMem, &bufDesc));

    m_importedBuffers[m_importedCount].extMemory = extMem;
    m_importedBuffers[m_importedCount].devicePtr = devicePtr;
    m_importedBuffers[m_importedCount].size = size;
    m_importedCount++;

    fprintf(stderr, "[CUDA] Imported buffer: %zu bytes -> %p\n", size, devicePtr);
    return devicePtr;
}

void CudaInterop::importTimelineSemaphore(HANDLE semaphoreHandle) {
    cudaExternalSemaphoreHandleDesc semDesc{};
    semDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    semDesc.handle.win32.handle = semaphoreHandle;
    semDesc.flags = 0;

    cudaExternalSemaphore_t extSem;
    CUDA_CHECK(cudaImportExternalSemaphore(&extSem, &semDesc));

    m_extSemaphore = extSem;
    fprintf(stderr, "[CUDA] Imported timeline semaphore\n");
}

void CudaInterop::waitSemaphore(uint64_t value) {
    if (!m_extSemaphore) return;

    cudaExternalSemaphoreWaitParams params{};
    params.params.fence.value = value;
    params.flags = 0;

    cudaExternalSemaphore_t sem = static_cast<cudaExternalSemaphore_t>(m_extSemaphore);
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(
        &sem, &params, 1, static_cast<cudaStream_t>(m_stream)));
}

void CudaInterop::signalSemaphore(uint64_t value) {
    if (!m_extSemaphore) return;

    cudaExternalSemaphoreSignalParams params{};
    params.params.fence.value = value;
    params.flags = 0;

    cudaExternalSemaphore_t sem = static_cast<cudaExternalSemaphore_t>(m_extSemaphore);
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(
        &sem, &params, 1, static_cast<cudaStream_t>(m_stream)));
}

bool CudaInterop::syncStream() {
    if (!m_stream) return true;
    cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(m_stream));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Stream sync error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

void* CudaInterop::stream() const {
    return m_stream;
}

} // namespace cosmico
