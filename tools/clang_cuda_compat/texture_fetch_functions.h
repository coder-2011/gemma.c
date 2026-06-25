#pragma once

#if CUDA_VERSION >= 12000
// Clang 18 still includes this removed CUDA texture header while setting up
// CUDA 12 wrappers. This lint-only shim keeps legacy texture references parsable.
template <class T, int texType, enum cudaTextureReadMode mode>
struct texture {};
#endif
