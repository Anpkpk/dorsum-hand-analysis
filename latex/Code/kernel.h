#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernel_MIP_Projection(unsigned char* d_volume, unsigned char* d_mip, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        unsigned char max_val = 0;

        for (int z = 0; z < D; z++) {
            long long idx = (long long)z * W * H + (long long)y * W + x;

            if (d_volume[idx] > max_val) {
                max_val = d_volume[idx];
            }
        }
        d_mip[y * W + x] = max_val;
    }
}

__global__ void kernel_Compute_GLCM(unsigned char* d_mip, int* d_glcm, int W, int H, int dx, int dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W - dx && y < H - dy) {
        int pixel_val1 = d_mip[y * W + x];

        int pixel_val2 = d_mip[(y + dy) * W + (x + dx)];

        int glcm_idx = pixel_val1 * 256 + pixel_val2;

        atomicAdd(&d_glcm[glcm_idx], 1);
    }
}


__global__ void kernel_Mean_Blur(unsigned char* d_mip, unsigned char* d_blurred, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < W - 1 && y >= 1 && y < H - 1) {
        int sum = 0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_idx = (y + dy) * W + (x + dx);
                sum += d_mip[neighbor_idx];
            }
        }

        d_blurred[y * W + x] = (unsigned char)(sum / 9);
    }
}

__global__ void kernel_Thresholding(unsigned char* d_blurred, unsigned char* d_binary, int W, int H, int threshold_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;

        if (d_blurred[idx] > threshold_value) {
            d_binary[idx] = 255;
        } else {
            d_binary[idx] = 0; 
        }
    }
}

__global__ void kernel_Count_Vessels(unsigned char* d_binary, int* d_white_count, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;

        if (d_binary[idx] == 255) {
            atomicAdd(d_white_count, 1);
        }
    }
}