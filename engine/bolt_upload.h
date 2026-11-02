// Checkpoint bytes to the GPU through a pinned staging buffer.
//
// Every tensor in an mmap'd safetensors shard starts after a header of any length, so its
// address is almost never aligned. A pageable copy from such an address takes a slow path in
// the driver: on an RTX Spark N1X (driver 615.78.08) the whole checkpoint took 27 minutes this
// way, against seconds once each piece is first copied into pinned memory. The CPU copy out of
// the page cache costs a few seconds over the 23 GB.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>

inline cudaError_t bolt_upload(void* dst, const void* src, size_t n) {
    static void* pinned = nullptr;
    static const size_t capacity = (size_t)64 << 20;
    if (!pinned) {
        cudaError_t e = cudaMallocHost(&pinned, capacity);
        if (e != cudaSuccess) { pinned = nullptr; return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice); }
    }
    for (size_t off = 0; off < n; off += capacity) {
        size_t len = n - off < capacity ? n - off : capacity;
        memcpy(pinned, (const char*)src + off, len);
        cudaError_t e = cudaMemcpy((char*)dst + off, pinned, len, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}
