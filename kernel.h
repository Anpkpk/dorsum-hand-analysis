#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ====================================================================
// Bước 1: Ép khối 3D thành khối 2D (MIP Projection)
// ====================================================================

// Đầu vào: d_volume (Khối 3D gốc)
// Đầu ra: d_mip (Tấm ảnh 2D chứa mạng lưới gân máu)

// ====================================================================
// BƯỚC 1 MỚI: Ép khối 3D thành ảnh 2D bề mặt (En-face MIP)
// ====================================================================
// Đầu vào: Khối 3D
// Đầu ra: Bản đồ mạch máu kích thước H x D (H x D)
__global__ void kernel_Enface_MIP(float* d_volume, float* d_enface, int W, int H, int D) {
    // Tọa độ mặt phẳng nhìn từ trên xuống lúc này là Y-Z (Height và Depth)
    int y = blockIdx.x * blockDim.x + threadIdx.x; // H
    int z = blockIdx.y * blockDim.y + threadIdx.y; // D

    if (y < H && z < D) {
        float max_val = 0;
        
        // Tia sáng đâm xuyên theo chiều sâu X (Width = 560)
        for (int x = 0; x < W; x++) {
            long long idx = (long long)z * W * H + (long long)y * W + x;
            if (d_volume[idx] > max_val) {
                max_val = d_volume[idx];
            }
        }
        // Ghi lên tấm ảnh bề mặt có kích thước H x D
        d_enface[z * H + y] = max_val;
    }
}

__global__ void kernel_MIP_Projection(float* d_volume, float* d_mip, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        float max_val = 0;

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

__global__ void kernel_Compute_GLCM(float* d_mip, int* d_glcm, int W, int H, int dx, int dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Đảm bảo không quét vượt quá lề của bức ảnh
    if (x < W - dx && y < H - dy) {
        // Lấy giá trị của điểm ảnh hiện tại (pixel 1)
        int pixel_val1 = min(max((int)d_mip[y * W + x], 0), 255);
        int pixel_val2 = min(max((int)d_mip[(y+dy)*W + (x+dx)], 0), 255);

        // Tính toán tọa độ của cặp này trong ma trận 256x256
        int glcm_idx = pixel_val1 * 256 + pixel_val2;

        // Các Thread tranh nhau cộng 1 vào ô này (Phải dùng atomicAdd để an toàn)
        atomicAdd(&d_glcm[glcm_idx], 1);
    }
}

// __global__ void kernel_Compute_GLCM(float* d_mip, int* d_glcm, int W, int H, int dx, int dy) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     // 1. Đảm bảo pixel gốc nằm gọn trong giới hạn của bức ảnh
//     if (x < W && y < H) {
        
//         // 2. Tính tọa độ của pixel lân cận theo hướng (dx, dy)
//         int nx = x + dx;
//         int ny = y + dy;

//         // 3. Kiểm tra xem pixel lân cận có nằm trong ảnh không 
//         // (Kiểm tra cả cận dưới >= 0 và cận trên < W, H để bắt các trường hợp dx, dy bị âm)
//         if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
            
//             // Lấy giá trị của điểm ảnh hiện tại (pixel 1) và lân cận (pixel 2)
//             int pixel_val1 = min(max((int)d_mip[y * W + x], 0), 255);
//             int pixel_val2 = min(max((int)d_mip[ny * W + nx], 0), 255);

//             // Tính toán tọa độ của cặp này trong ma trận 256x256
//             int glcm_idx = pixel_val1 * 256 + pixel_val2;

//             // Các Thread tranh nhau cộng 1 vào ô này (Phải dùng atomicAdd để an toàn)
//             atomicAdd(&d_glcm[glcm_idx], 1);
//         }
//     }
// }

// ====================================================================
// Bước 2: Làm mờ giảm nhiễu (Mean Filter)
// ====================================================================

// Đầu vào: d_mip (Ảnh 2D vừa ép xong)
// Đầu ra: d_blurred (Ảnh 2D đã làm mịn)
__global__ void kernel_Mean_Blur(float* d_mip, float* d_blurred, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bỏ qua viền ngoài cùng để tránh lỗi out-of-bounds
    if (x >= 1 && x < W - 1 && y >= 1 && y < H - 1) {
        float sum = 0;

        // Quét ma trận 3x3
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_idx = (y + dy) * W + (x + dx);
                sum += d_mip[neighbor_idx];
            }
        }

        // Chia 9 để lấy trung bình
        d_blurred[y * W + x] = sum / 9.0f;
    }
}

// ====================================================================
// Bước 3: Phân ngưỡng (Thresholding) - Trắng/đen hóa
// ====================================================================

// Đầu vào: d_blurred
// Đầu ra: d_binary (Ảnh trắng đen)
__global__ void kernel_Thresholding(float* d_blurred, float* d_binary, int W, int H, float threshold_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;

        if (d_blurred[idx] > threshold_value) {
            d_binary[idx] = 255; // Mạch máu -> Màu đen 
        } else {
            d_binary[idx] = 0;   // Da/Nền -> Màu trắng
        }
    }
}

// ====================================================================
// Bước 4: Vessel Analysis (Đếm mật độ mạch máu)
// ====================================================================

// Đầu vào: d_binary (Ảnh trắng đen)
// Đầu ra: d_white_count (Một biến duy nhất lưu tổng số pixel trắng)
__global__ void kernel_Count_Vessels(float* d_binary, int* d_white_count, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int idx = y * W + x;

        // Nếu pixel là màu đen (mạch máu)
        if (d_binary[idx] == 0) {
            atomicAdd(d_white_count, 1); // Tranh nhau cộng tiền vào quỹ chung
        }
    }
}