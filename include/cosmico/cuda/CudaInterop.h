#pragma once
#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cosmico {

class CudaInterop {
public:
    void init();
    void destroy();

    // Import a Vulkan buffer's memory via Win32 handle
    // Returns a CUDA device pointer to the shared memory
    void* importBuffer(HANDLE memoryHandle, size_t size);

    // Import a Vulkan timeline semaphore via Win32 handle
    void importTimelineSemaphore(HANDLE semaphoreHandle);

    // Wait for semaphore to reach a value (from CUDA side)
    void waitSemaphore(uint64_t value);

    // Signal semaphore to a value (from CUDA side)
    void signalSemaphore(uint64_t value);

    // Synchronize the CUDA stream and check for errors (returns true on success)
    bool syncStream();

    // Get the CUDA stream for kernel launches
    void* stream() const;

    bool isInitialized() const { return m_initialized; }

private:
    void* m_stream = nullptr;         // cudaStream_t
    void* m_extSemaphore = nullptr;   // cudaExternalSemaphore_t
    bool m_initialized = false;

    struct ImportedMemory {
        void* extMemory;   // cudaExternalMemory_t
        void* devicePtr;   // mapped device pointer
        size_t size;
    };
    static constexpr int MAX_IMPORTED_BUFFERS = 4;
    ImportedMemory m_importedBuffers[MAX_IMPORTED_BUFFERS] = {};
    int m_importedCount = 0;
};

} // namespace cosmico
