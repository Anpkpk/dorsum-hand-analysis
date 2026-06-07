#include <iostream>
#include <stdio.h>
#include <math.h>
#include "kernel.h"

int main() {
    // 1. THÔNG SỐ CƠ BẢN
    int W = 560;  // Trục đâm sâu vào da
    int H = 1000; // Trục ngang 1
    int D = 500;  // Trục ngang 2
    
    long long total_pixels = (long long)W * H * D;
    
    // RẤT QUAN TRỌNG: Ảnh bề mặt lúc này có kích thước H x D (1000 x 500)
    long long enface_pixels = (long long)H * D; 

    // 2. KHAI BÁO & CẤP PHÁT BỘ NHỚ TRÊN GPU (VRAM)
    float *d_volume, *d_enface, *d_blurred, *d_binary;
    int *d_white_count, *d_glcm;

    cudaMalloc(&d_volume, total_pixels * sizeof(float));
    cudaMalloc(&d_enface, enface_pixels * sizeof(float));
    cudaMalloc(&d_blurred, enface_pixels * sizeof(float));
    cudaMalloc(&d_binary, enface_pixels * sizeof(float));
    cudaMalloc(&d_white_count, sizeof(int));
    cudaMalloc(&d_glcm, 256 * 256 * sizeof(int));

    // ====================================================================
    // ĐỌC DỮ LIỆU FILE RAW LÊN GPU
    // ====================================================================
    float* h_volume = new float[total_pixels];
    printf("Dang doc file volume_3d_o.raw vao RAM...\n");
    FILE *f = fopen("volume_3d_o.raw", "rb");
    if (f == NULL) {
        printf("LOI: Khong tim thay file. Kiem tra lai thu muc!\n");
        return -1;
    }
    fread(h_volume, sizeof(float), total_pixels, f);
    fclose(f);

    printf("Dang day du lieu len GPU...\n");
    cudaMemcpy(d_volume, h_volume, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_volume;

    // ====================================================================
    // CẤU HÌNH LƯỚI THỢ CHO MẶT PHẲNG EN-FACE (H x D)
    // ====================================================================
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((H + 15) / 16, (D + 15) / 16); // Thay W bằng H

    // ====================================================================
    // BƯỚC MỞ MÀN: ÉP KHỐI 3D THÀNH ẢNH EN-FACE
    // ====================================================================
    printf("\nDang ep khoi 3D xuong mat phang En-face (Truc X)...\n");
    kernel_Enface_MIP<<<blocksPerGrid, threadsPerBlock>>>(d_volume, d_enface, W, H, D);
    cudaDeviceSynchronize();

    // ====================================================================
    // NHÁNH 1: VESSEL ANALYSIS (PHÂN TÍCH MẠCH MÁU)
    // ====================================================================
    printf("\n--- NHANH 1: PHAN TICH MAT DO MACH MAU ---\n");
    cudaMemset(d_blurred, 0, enface_pixels * sizeof(float));
    
    // Đổi tham số thành H, D
    kernel_Mean_Blur<<<blocksPerGrid, threadsPerBlock>>>(d_enface, d_blurred, H, D);
    cudaDeviceSynchronize();

    // Bạn có thể chỉnh số 42 ở đây tùy vào độ sáng của ảnh mạch máu mới
    kernel_Thresholding<<<blocksPerGrid, threadsPerBlock>>>(d_blurred, d_binary, H, D, 45.0f);
    cudaDeviceSynchronize();

    cudaMemset(d_white_count, 0, sizeof(int));
    kernel_Count_Vessels<<<blocksPerGrid, threadsPerBlock>>>(d_binary, d_white_count, H, D);
    cudaDeviceSynchronize();

    int h_white_count = 0;
    cudaMemcpy(&h_white_count, d_white_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("=> Mat do mach mau: %.2f%%\n", (float)h_white_count / enface_pixels * 100.0f);

    // ====================================================================
    // NHÁNH 2: TEXTURE ANALYSIS (GLCM) TRÊN ẢNH EN-FACE
    // ====================================================================
    printf("\n--- NHANH 2: PHAN TICH KET CAU MO (GLCM) ---\n");
    cudaMemset(d_glcm, 0, 256 * 256 * sizeof(int));

    kernel_Compute_GLCM<<<blocksPerGrid, threadsPerBlock>>>(d_enface, d_glcm, H, D, 0, 1); 
    cudaDeviceSynchronize();

    int h_glcm[256 * 256];
    cudaMemcpy(h_glcm, d_glcm, 256 * 256 * sizeof(int), cudaMemcpyDeviceToHost);

    double total_pairs = 0;
    for (int i = 0; i < 256 * 256; i++) total_pairs += h_glcm[i];

    double contrast = 0.0, homogeneity = 0.0, energy = 0.0;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            double p = h_glcm[i * 256 + j] / total_pairs;
            if (p > 0) {
                contrast += p * (i - j) * (i - j);
                homogeneity += p / (1.0 + abs(i - j));
                energy += p * p;
            }
        }
    }
    printf("=> Contrast    : %f\n", contrast);
    printf("=> Homogeneity : %f\n", homogeneity);
    printf("=> Energy      : %f\n", energy);

    // ====================================================================
    // LƯU ẢNH KẾT QUẢ
    // ====================================================================
    printf("\nDang xuat anh ket qua ra file RAW...\n");
    float* h_enface_out = new float[enface_pixels];
    float* h_blurred_out = new float[enface_pixels];
    float* h_binary_out = new float[enface_pixels];

    cudaMemcpy(h_enface_out, d_enface, enface_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blurred_out, d_blurred, enface_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_binary_out, d_binary, enface_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f_enface = fopen("ketqua_1_Enface.raw", "wb");
    if(f_enface) { fwrite(h_enface_out, sizeof(float), enface_pixels, f_enface); fclose(f_enface); }

    FILE* f_blur = fopen("ketqua_2_Blur.raw", "wb");
    if(f_blur) { fwrite(h_blurred_out, sizeof(float), enface_pixels, f_blur); fclose(f_blur); }

    FILE* f_binary = fopen("ketqua_3_Binary.raw", "wb");
    if(f_binary) { fwrite(h_binary_out, sizeof(float), enface_pixels, f_binary); fclose(f_binary); }

    delete[] h_enface_out; delete[] h_blurred_out; delete[] h_binary_out;

    printf("Da luu thanh cong 3 file: ketqua_1_Enface.raw, ketqua_2_Blur.raw, ketqua_3_Binary.raw\n");
    printf("=> Hay mo bang ImageJ voi thong so: Width = %d, Height = %d, 8-bit!\n\n", H, D);

    cudaFree(d_volume); 
    cudaFree(d_enface); 
    cudaFree(d_blurred); 
    cudaFree(d_binary); 
    cudaFree(d_white_count); 
    cudaFree(d_glcm);

    return 0;
}