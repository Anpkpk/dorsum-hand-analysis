#include <iostream>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include "kernel.h"

// ====================================================================
// Bước 1: Ép khối 3D thành ảnh 2D bề mặt (En-face MIP) - CPU
// ====================================================================
// Đầu vào: h_volume (Khối 3D gốc)
// Đầu ra:  h_enface (Bản đồ mạch máu kích thước H x D)
void cpu_Enface_MIP(float* h_volume, float* h_enface, int W, int H, int D) {
    for (int y = 0; y < H; y++) {       // Trục H
        for (int z = 0; z < D; z++) {   // Trục D
            float max_val = 0;

            // Tia sáng đâm xuyên theo chiều sâu X (Width = 560)
            for (int x = 0; x < W; x++) {
                long long idx = (long long)z * W * H + (long long)y * W + x;
                if (h_volume[idx] > max_val) {
                    max_val = h_volume[idx];
                }
            }
            // Ghi lên tấm ảnh bề mặt có kích thước H x D
            h_enface[z * H + y] = max_val;
        }
    }
}

// ====================================================================
// Bước 2: Xây dựng ma trận GLCM - CPU
// ====================================================================
// Đầu vào: h_mip  (Ảnh 2D xám gốc)
// Đầu ra:  h_glcm (Mảng 1D mô phỏng ma trận 256x256, khởi tạo toàn số 0)
// dx, dy: Khoảng cách giữa 2 điểm ảnh muốn so sánh
void cpu_Compute_GLCM(float* h_mip, int* h_glcm, int W, int H, int dx, int dy) {
    for (int y = 0; y < H - dy; y++) {
        for (int x = 0; x < W - dx; x++) {

            // Đảm bảo không quét vượt quá lề của bức ảnh
            // Lấy giá trị của điểm ảnh hiện tại (pixel 1)
            int pixel_val1 = min(max((int)h_mip[y * W + x], 0), 255);
            int pixel_val2 = min(max((int)h_mip[(y+dy)*W + (x+dx)], 0), 255);

            // Tính toán tọa độ của cặp này trong ma trận 256x256
            int glcm_idx = pixel_val1 * 256 + pixel_val2;

            // CPU chạy tuần tự nên không cần atomicAdd
            h_glcm[glcm_idx]++;
        }
    }
}

// ====================================================================
// Bước 2: Làm mờ giảm nhiễu (Mean Filter) - CPU
// ====================================================================
// Đầu vào: h_mip     (Ảnh 2D vừa ép xong)
// Đầu ra:  h_blurred (Ảnh 2D đã làm mịn)
void cpu_Mean_Blur(float* h_mip, float* h_blurred, int W, int H) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {

            // Bỏ qua viền ngoài cùng để tránh lỗi out-of-bounds
            if (x >= 1 && x < W - 1 && y >= 1 && y < H - 1) {
                float sum = 0;

                // Quét ma trận 3x3
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int neighbor_idx = (y + dy) * W + (x + dx);
                        sum += h_mip[neighbor_idx];
                    }
                }
                // Chia 9 để lấy trung bình
                h_blurred[y * W + x] = sum / 9.0f;
            }
        }
    }
}

// ====================================================================
// Bước 3: Phân ngưỡng (Thresholding) - Trắng/đen hóa - CPU
// ====================================================================
// Đầu vào: h_blurred
// Đầu ra:  h_binary (Ảnh trắng đen)
void cpu_Thresholding(float* h_blurred, float* h_binary, int W, int H, float threshold_value) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;

            if (h_blurred[idx] > threshold_value) {
                h_binary[idx] = 255; // Mạch máu -> Màu trắng
            } else {
                h_binary[idx] = 0;   // Da/Nền  -> Màu đen
            }
        }
    }
}

// ====================================================================
// Bước 4: Vessel Analysis (Đếm mật độ mạch máu) - CPU
// ====================================================================
// Đầu vào: h_binary      (Ảnh trắng đen)
// Đầu ra:  h_white_count (Một biến duy nhất lưu tổng số pixel trắng)
void cpu_Count_Vessels(float* h_binary, int* h_white_count, int W, int H) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;

            // Nếu pixel là màu trắng (mạch máu)
            if (h_binary[idx] == 255) {
                (*h_white_count)++; // CPU tuần tự nên cộng thẳng, không cần atomicAdd
            }
        }
    }
}

// ============================================================
// HÀM IN KẾT QUẢ PHÂN TÍCH
// ============================================================
void print_analysis_results(int white_count, long long enface_pixels,
                             int* glcm, const char* label) {
    printf("\n========== KET QUA: %s ==========\n", label);
    printf("=> Mat do mach mau: %.2f%%\n",
           (float)white_count / enface_pixels * 100.0f);

    double total_pairs = 0;
    for (int i = 0; i < 256 * 256; i++) total_pairs += glcm[i];

    double contrast = 0.0, homogeneity = 0.0, energy = 0.0;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            double p = glcm[i * 256 + j] / total_pairs;
            if (p > 0) {
                contrast    += p * (i - j) * (i - j);
                homogeneity += p / (1.0 + abs(i - j));
                energy      += p * p;
            }
        }
    }
    printf("=> Contrast    : %f\n", contrast);
    printf("=> Homogeneity : %f\n", homogeneity);
    printf("=> Energy      : %f\n", energy);
}

int main() {
    // ====================================================
    // 1. THÔNG SỐ CƠ BẢN
    // ====================================================
    int W = 560, H = 1000, D = 500;
    long long total_pixels  = (long long)W * H * D;
    long long enface_pixels = (long long)H * D;

    // ====================================================
    // ĐỌC DỮ LIỆU FILE RAW VÀO RAM (dùng chung 2 nhánh)
    // ====================================================
    float* h_volume = new float[total_pixels];
    printf("Dang doc file volume_3d_o.raw vao RAM...\n");
    FILE* f = fopen("volume_3d_o.raw", "rb");
    if (!f) { printf("LOI: Khong tim thay file!\n"); return -1; }
    fread(h_volume, sizeof(float), total_pixels, f);
    fclose(f);

    // Cấp phát bộ nhớ host dùng chung
    float* h_enface  = new float[enface_pixels];
    float* h_blurred = new float[enface_pixels];
    float* h_binary  = new float[enface_pixels];
    int*   h_glcm    = new int[256 * 256];

    // ====================================================
    //  NHÁNH A: THỰC THI TRÊN CPU
    // ====================================================
    printf("\n============================================================\n");
    printf("  NHANH A: THUC THI TREN CPU\n");
    printf("============================================================\n");

    auto cpu_start = std::chrono::high_resolution_clock::now();

    // --- A1: En-face MIP ---
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_Enface_MIP(h_volume, h_enface, W, H, D);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[CPU] Enface MIP    : %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // --- A2: Mean Blur ---
    memset(h_blurred, 0, enface_pixels * sizeof(float));
    t0 = std::chrono::high_resolution_clock::now();
    cpu_Mean_Blur(h_enface, h_blurred, H, D);
    t1 = std::chrono::high_resolution_clock::now();
    printf("[CPU] Mean Blur     : %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // --- A3: Thresholding ---
    t0 = std::chrono::high_resolution_clock::now();
    cpu_Thresholding(h_blurred, h_binary, H, D, 45.0f);
    t1 = std::chrono::high_resolution_clock::now();
    printf("[CPU] Thresholding  : %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // --- A4: Count Vessels ---
    int cpu_white_count = 0;
    t0 = std::chrono::high_resolution_clock::now();
    cpu_Count_Vessels(h_binary, &cpu_white_count, H, D);
    t1 = std::chrono::high_resolution_clock::now();
    printf("[CPU] Count Vessels : %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // --- A5: GLCM ---
    memset(h_glcm, 0, 256 * 256 * sizeof(int));
    t0 = std::chrono::high_resolution_clock::now();
    cpu_Compute_GLCM(h_enface, h_glcm, H, D, 0, 1);
    t1 = std::chrono::high_resolution_clock::now();
    printf("[CPU] GLCM          : %.3f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_total_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("--------------------------------------------------\n");
    printf("[CPU] TONG THOI GIAN: %.3f ms  (%.3f giay)\n",
           cpu_total_ms, cpu_total_ms / 1000.0);

    print_analysis_results(cpu_white_count, enface_pixels, h_glcm, "CPU");

    // ====================================================
    //  NHÁNH B: THỰC THI TRÊN GPU
    // ====================================================
    printf("\n============================================================\n");
    printf("  NHANH B: THUC THI TREN GPU\n");
    printf("============================================================\n");

    float *d_volume, *d_enface, *d_blurred, *d_binary;
    int   *d_white_count, *d_glcm;
    cudaMalloc(&d_volume,      total_pixels  * sizeof(float));
    cudaMalloc(&d_enface,      enface_pixels * sizeof(float));
    cudaMalloc(&d_blurred,     enface_pixels * sizeof(float));
    cudaMalloc(&d_binary,      enface_pixels * sizeof(float));
    cudaMalloc(&d_white_count, sizeof(int));
    cudaMalloc(&d_glcm,        256 * 256 * sizeof(int));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float gpu_ms = 0.0f, gpu_total_ms = 0.0f;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((H + 15) / 16, (D + 15) / 16);

    // --- B0: Copy H→D ---
    cudaEventRecord(ev_start);
    cudaMemcpy(d_volume, h_volume, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] H->D Transfer : %.3f ms\n", gpu_ms);
    gpu_total_ms += gpu_ms;

    // --- B1: Enface MIP ---
    cudaEventRecord(ev_start);
    kernel_Enface_MIP<<<blocksPerGrid, threadsPerBlock>>>(d_volume, d_enface, W, H, D);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] Enface MIP    : %.3f ms\n", gpu_ms);
    gpu_total_ms += gpu_ms;

    // --- B2: Mean Blur ---
    cudaMemset(d_blurred, 0, enface_pixels * sizeof(float));
    cudaEventRecord(ev_start);
    kernel_Mean_Blur<<<blocksPerGrid, threadsPerBlock>>>(d_enface, d_blurred, H, D);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] Mean Blur     : %.3f ms\n", gpu_ms);
    gpu_total_ms += gpu_ms;

    // --- B3: Thresholding ---
    cudaEventRecord(ev_start);
    kernel_Thresholding<<<blocksPerGrid, threadsPerBlock>>>(d_blurred, d_binary, H, D, 45.0f);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] Thresholding  : %.3f ms\n", gpu_ms);
    gpu_total_ms += gpu_ms;

    // --- B4: Count Vessels ---
    cudaMemset(d_white_count, 0, sizeof(int));
    cudaEventRecord(ev_start);
    kernel_Count_Vessels<<<blocksPerGrid, threadsPerBlock>>>(d_binary, d_white_count, H, D);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] Count Vessels : %.3f ms\n", gpu_ms);
    gpu_total_ms += gpu_ms;

    // --- B5: GLCM ---
    cudaMemset(d_glcm, 0, 256 * 256 * sizeof(int));
    cudaEventRecord(ev_start);
    kernel_Compute_GLCM<<<blocksPerGrid, threadsPerBlock>>>(d_enface, d_glcm, H, D, 0, 1);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] GLCM          : %.3f ms\n", gpu_ms);
    gpu_total_ms += gpu_ms;

    // --- B6: Copy D→H ---
    int h_white_count_gpu = 0;
    int h_glcm_gpu[256 * 256];
    cudaEventRecord(ev_start);
    cudaMemcpy(&h_white_count_gpu, d_white_count, sizeof(int),         cudaMemcpyDeviceToHost);
    cudaMemcpy(h_glcm_gpu,         d_glcm,        256*256*sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    printf("[GPU] D->H Transfer : %.3f ms\n", gpu_ms);
    gpu_total_ms += gpu_ms;

    printf("--------------------------------------------------\n");
    printf("[GPU] TONG THOI GIAN: %.3f ms  (%.3f giay)\n",
           gpu_total_ms, gpu_total_ms / 1000.0);

    print_analysis_results(h_white_count_gpu, enface_pixels, h_glcm_gpu, "GPU");

    // ====================================================
    //  BẢNG SO SÁNH TỔNG KẾT
    // ====================================================
    printf("\n============================================================\n");
    printf("  BANG SO SANH CPU vs GPU\n");
    printf("============================================================\n");
    printf("  CPU tong thoi gian : %10.3f ms\n", cpu_total_ms);
    printf("  GPU tong thoi gian : %10.3f ms\n", gpu_total_ms);
    printf("  He so tang toc     : %10.2fx\n",   cpu_total_ms / gpu_total_ms);
    printf("============================================================\n\n");

    // ====================================================
    //  LƯU FILE KẾT QUẢ (dùng output từ GPU)
    // ====================================================
    printf("Dang xuat anh ket qua ra file RAW...\n");
    float* h_enface_out  = new float[enface_pixels];
    float* h_blurred_out = new float[enface_pixels];
    float* h_binary_out  = new float[enface_pixels];

    cudaMemcpy(h_enface_out,  d_enface,  enface_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blurred_out, d_blurred, enface_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_binary_out,  d_binary,  enface_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* fout;
    fout = fopen("ketqua_1_Enface.raw", "wb");
    if(fout) { fwrite(h_enface_out,  sizeof(float), enface_pixels, fout); fclose(fout); }
    fout = fopen("ketqua_2_Blur.raw", "wb");
    if(fout) { fwrite(h_blurred_out, sizeof(float), enface_pixels, fout); fclose(fout); }
    fout = fopen("ketqua_3_Binary.raw", "wb");
    if(fout) { fwrite(h_binary_out,  sizeof(float), enface_pixels, fout); fclose(fout); }

    printf("Da luu 3 file: ketqua_1_Enface.raw, ketqua_2_Blur.raw, ketqua_3_Binary.raw\n");
    printf("=> Mo bang ImageJ: Width=%d, Height=%d, 32-bit float\n", H, D);

    // ====================================================
    //  DỌN DẸP BỘ NHỚ
    // ====================================================
    delete[] h_volume;  delete[] h_enface;  delete[] h_blurred;
    delete[] h_binary;  delete[] h_glcm;
    delete[] h_enface_out; delete[] h_blurred_out; delete[] h_binary_out;

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaFree(d_volume);  cudaFree(d_enface);  cudaFree(d_blurred);
    cudaFree(d_binary);  cudaFree(d_white_count);  cudaFree(d_glcm);

    return 0;
}