import torch
from torch.utils.cpp_extension import load_inline
import subprocess
import shutil

def _check_ninja():
    """Check if ninja is available in the system path."""
    return shutil.which("ninja") is not None

cuda_source = """
#include <torch/extension.h>
#include <cuda.hpp>
#include <cuda_runtime.hpp>

__global__ void bone_enhancement_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    float wavelet_gain,
    float ridge_alpha) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (x < width && y < height) {
        int idx = b * (height * width) + y * width + x;
        
        // 1. Local 3x3 Gradient Calculation (Sobel-like)
        // Using registers for speed
        float val_m1_m1 = (x > 0 && y > 0) ? input[idx - width - 1] : input[idx];
        float val_0_m1  = (y > 0) ? input[idx - width] : input[idx];
        float val_p1_m1 = (x < width - 1 && y > 0) ? input[idx - width + 1] : input[idx];
        
        float val_m1_0  = (x > 0) ? input[idx - 1] : input[idx];
        float val_0_0   = input[idx];
        float val_p1_0  = (x < width - 1) ? input[idx + 1] : input[idx];
        
        float val_m1_p1 = (x > 0 && y < height - 1) ? input[idx + width - 1] : input[idx];
        float val_0_p1  = (y < height - 1) ? input[idx + width] : input[idx];
        float val_p1_p1 = (x < width - 1 && y < height - 1) ? input[idx + width + 1] : input[idx];

        // 2. Fused Hessian Calculation (Second Derivatives)
        float ixx = (val_m1_0 - 2.0f * val_0_0 + val_p1_0);
        float iyy = (val_0_m1 - 2.0f * val_0_0 + val_0_p1);
        float ixy = 0.25f * (val_p1_p1 - val_m1_p1 - val_p1_m1 + val_m1_m1);

        // 3. Ridge Detection via Eigenvalues (Frangi Approximation)
        // Solve characteristic equation: det(H - lambda*I) = 0
        float trace = ixx + iyy;
        float det = ixx * iyy - ixy * ixy;
        float disc = sqrtf(fmaxf(0.0f, trace * trace / 4.0f - det));
        float lambda1 = trace / 2.0f + disc;
        float lambda2 = trace / 2.0f - disc;

        // Fracture lines are usually high negative eigenvalues (ridges)
        float ridge = ridge_alpha * fmaxf(0.0f, -lambda1) * fmaxf(0.0f, -lambda2);
        
        // 4. Wavelet-like Detail Boost (High Pass)
        float high_pass = val_0_0 - (val_m1_m1 + val_0_m1 + val_p1_m1 + val_m1_0 + val_p1_0 + val_m1_p1 + val_0_p1 + val_p1_p1) / 8.0f;
        float boost = high_pass * wavelet_gain;

        // 5. Output Fusion (Original + Boost + Ridge)
        output[idx] = val_0_0 + boost + ridge;
    }
}

torch::Tensor bone_enhance_cuda(torch::Tensor input, float wavelet_gain, float ridge_alpha) {
    auto output = torch::zeros_like(input);
    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);

    const int threads = 16;
    dim3 blocks((width + threads - 1) / threads, (height + threads - 1) / threads, batch_size);
    dim3 threads_per_block(threads, threads);

    bone_enhancement_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, 1, height, width,
        wavelet_gain, ridge_alpha
    );

    return output;
}
"""

cpp_source = "torch::Tensor bone_enhance_cuda(torch::Tensor input, float wavelet_gain, float ridge_alpha);"

def _check_ninja():
    """Ninja is no longer strictly required since we use with_ninja=False."""
    return torch.cuda.is_available()

class FusedBoneCUDA:
    _module = None
    _failed = False

    @classmethod
    def get_module(cls):
        if cls._failed:
            return None
        if cls._module is None:
            if not _check_ninja():
                cls._failed = True
                return None

            try:
                print("Compiling Fused CUDA Kernels... (This may take a minute on first run)")
                cls._module = load_inline(
                    name="bone_cuda_ops",
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=['bone_enhance_cuda'],
                    verbose=False
                )

            except Exception as e:
                print(f"Error compiling CUDA kernels: {e}")
                print("Falling back to standard PyTorch GPU operations.")
                cls._failed = True
                return None
        return cls._module

def apply_fused_bone_enhancement(tensor, wavelet_gain=2.0, ridge_alpha=5.0):
    """
    Apply high-performance CUDA kernel for fracture enhancement.
    Input: B x 1 x H x W (Float Tensor on GPU)
    """
    if not tensor.is_cuda:
        return tensor # Fallback
    
    module = FusedBoneCUDA.get_module()
    if module is None:
        return None # Signal fallback required
        
    return module.bone_enhance_cuda(tensor, wavelet_gain, ridge_alpha)
