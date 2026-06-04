#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ====================================================================
// Bước 1: Ép khối 3D thành khối 2D (MIP Projection)
// ====================================================================

// Đầu vào: d_volume (Khối 3D gốc)
// Đầu ra: d_mip (Tấm ảnh 2D chứa mạng lưới gân máu)
__global__ void kernel_MIP_Projection(unsigned char* d_volume, unsigned char* d_mip, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        unsigned char max_val = 0;

        // Vòng lặp chạy dọc theo chiều sâu Z
        for (int z = 0; z < D; z++) {
            // Công thức tính index 1D cho mảng 3D
            long long idx = (long long)z * W * H + (long long)y * W + x;

            // Tìm điểm sáng nhất (mạch máu)
            if (d_volume[idx] > max_val) {
                max_val = d_volume[idx];
            }
        }
        // Ghi lên ảnh 2D
        d_mip[y * W + x] = max_val;
    }
}

// ====================================================================
// Bước 2: Xây dựng ma trận GLCM
// ====================================================================

// Đầu vào: d_mip (Ảnh 2D xám gốc, chưa qua làm mờ hay đen trắng)
// Đầu ra: d_glcm (Mảng 1D mô phỏng ma trận 256x256, khởi tạo toàn số 0)
// dx, dy: Khoảng cách giữa 2 điểm ảnh muốn so sánh (Thường dùng dx=1, dy=0 để so sánh điểm ảnh liền kề theo chiều ngang)

__global__ void kernel_Compute_GLCM(unsigned char* d_mip, int* d_glcm, int W, int H, int dx, int dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Đảm bảo không quét vượt quá lề của bức ảnh
    if (x < W - dx && y < H - dy) {
        // Lấy giá trị của điểm ảnh hiện tại (pixel 1)
        int pixel_val1 = d_mip[y * W + x];

        // Lấy giá trị của điểm ảnh hàng xóm (pixel 2)
        int pixel_val2 = d_mip[(y + dy) * W + (x + dx)];

        // Tính toán tọa độ của cặp này trong ma trận 256x256
        int glcm_idx = pixel_val1 * 256 + pixel_val2;

        // Các Thread tranh nhau cộng 1 vào ô này (Phải dùng atomicAdd để an toàn)
        atomicAdd(&d_glcm[glcm_idx], 1);
    }
}



// ====================================================================
// Bước 2: Làm mờ giảm nhiễu (Mean Filter)
// ====================================================================

// Đầu vào: d_mip (Ảnh 2D vừa ép xong)
// Đầu ra: d_blurred (Ảnh 2D đã làm mịn)
__global__ void kernel_Mean_Blur(unsigned char* d_mip, unsigned char* d_blurred, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bỏ qua viền ngoài cùng để tránh lỗi out-of-bounds
    if (x >= 1 && x < W - 1 && y >= 1 && y < H - 1) {
        int sum = 0;

        // Quét ma trận 3x3
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_idx = (y + dy) * W + (x + dx);
                sum += d_mip[neighbor_idx];
            }
        }

        // Chia 9 để lấy trung bình
        d_blurred[y * W + x] = (unsigned char)(sum / 9);
    }
}

// ====================================================================
// Bước 3: Phân ngưỡng (Thresholding) - Trắng/đen hóa
// ====================================================================

// Đầu vào: d_blurred
// Đầu ra: d_binary (Ảnh trắng đen)
__global__ void kernel_Thresholding(unsigned char* d_blurred, unsigned char* d_binary, int W, int H, int threshold_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;

        if (d_blurred[idx] > threshold_value) {
            d_binary[idx] = 255; // Mạch máu -> Màu trắng
        } else {
            d_binary[idx] = 0;   // Da/Nền -> Màu đen
        }
    }
}

// ====================================================================
// Bước 4: Vessel Analysis (Đếm mật độ mạch máu)
// ====================================================================

// Đầu vào: d_binary (Ảnh trắng đen)
// Đầu ra: d_white_count (Một biến duy nhất lưu tổng số pixel trắng)
__global__ void kernel_Count_Vessels(unsigned char* d_binary, int* d_white_count, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;

        // Nếu pixel là màu trắng (mạch máu)
        if (d_binary[idx] == 255) {
            atomicAdd(d_white_count, 1); // Tranh nhau cộng tiền vào quỹ chung
        }
    }
}