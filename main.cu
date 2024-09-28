#include <iostream>

static void _check_cuda_error(cudaError_t error_code, char const *const func,
                              const char *const file, int const line) {
    if (!error_code) {
        return;
    }

    std::cerr << "CUDA error at " << file << ": " << line << " '" << func << "'\n";
    auto error_str = cudaGetErrorString(error_code);
    std::cerr << "CUDA error " << static_cast<unsigned int>(error_code) << ": " << error_str
              << "\n";

    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(1);
}

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define CHECK_CUDA_ERROR(val) _check_cuda_error((val), #val, __FILE__, __LINE__)

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    printf("hello from CPU\n");

    cuda_hello<<<3,3>>>();
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return 0;
}
