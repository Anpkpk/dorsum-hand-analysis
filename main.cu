#include <iostream>
#include <stdio.h>
#include <math.h>
#include "kernel.h"

// int main() {
//     // 1. THÔNG SỐ CƠ BẢN (Sửa W, H, D theo dataset bạn chọn)
//     int W = 1000;
//     int H = 2048;
//     int D = 500;
//     long long total_pixels = (long long)W * H * D;
//     long long frame_pixels = (long long)W * H;

//     // 2. KHAI BÁO & CẤP PHÁT BỘ NHỚ TRÊN GPU (VRAM)
//     unsigned char *d_volume, *d_mip;
//     int *d_glcm;

//     cudaMalloc(&d_volume, total_pixels * sizeof(unsigned char));
//     cudaMalloc(&d_mip, frame_pixels * sizeof(unsigned char));
//     cudaMalloc(&d_glcm, 256 * 256 * sizeof(int)); // Bộ nhớ cho ma trận GLCM

//     // ====================================================================
//     // ĐOẠN ĐỌC DỮ LIỆU TỪ FILE .RAW VÀ ĐẨY LÊN GPU
//     // Cần chuyển đổi 500 file ảnh txt thành file volume_3d.raw
//     // Nếu không dùng cách này có thể tự thay đổi để đọc dữ liệu
//     // ====================================================================
//     // Cấp phát mảng tạm trên RAM (CPU)
//     unsigned char* h_volume = new unsigned char[total_pixels];

//     // Mở và đọc file nhị phân siêu tốc
//     printf("Dang doc file volume_3d.raw vao RAM...\n");
//     FILE *f = fopen("volume_3d.raw", "rb");
//     if (f == NULL) {
//         printf("LOI: Khong tim thay file volume_3d.raw. Kiem tra lai thu muc!\n");
//         return -1; // Dừng chương trình nếu không có file
//     }
//     fread(h_volume, sizeof(unsigned char), total_pixels, f);
//     fclose(f);

//     // Copy dữ liệu từ RAM sang GPU
//     printf("Dang day du lieu len GPU...\n");
//     cudaMemcpy(d_volume, h_volume, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

//     // Giải phóng RAM máy tính (Vì GPU đã giữ bản sao rồi)
//     delete[] h_volume;
//     // ====================================================================

//     // 3. CẤU HÌNH LƯỚI THỢ (Grid & Block)
//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((W + 15) / 16, (H + 15) / 16);

//     // ====================================================================
//     // BƯỚC MỞ MÀN: ÉP KHỐI 3D THÀNH ẢNH 2D
//     // ====================================================================
//     printf("Dang ep khoi 3D xuong 2D (MIP)...\n");
//     kernel_MIP_Projection<<<blocksPerGrid, threadsPerBlock>>>(d_volume, d_mip, W, H, D);
//     cudaDeviceSynchronize();
//     // ====================================================================
//     // NHÁNH 2: TEXTURE ANALYSIS (GLCM) - DÙNG ẢNH d_mip GỐC
//     // ====================================================================
//     printf("Dang chay Nhanh 2: Phan tich ket cau (GLCM)...\n");

//     // RẤT QUAN TRỌNG: Phải reset toàn bộ ma trận GLCM trên GPU về 0 trước khi đếm
//     cudaMemset(d_glcm, 0, 256 * 256 * sizeof(int));

//     // ĐÂY CHÍNH LÀ NƠI GỌI HÀM KERNEL GLCM (dx=1, dy=0)
//     kernel_Compute_GLCM<<<blocksPerGrid, threadsPerBlock>>>(d_mip, d_glcm, W, H, 1, 0);
//     cudaDeviceSynchronize();

//     // Mang ma trận GLCM từ GPU về CPU để tính toán Toán học
//     int h_glcm[256 * 256];
//     cudaMemcpy(h_glcm, d_glcm, 256 * 256 * sizeof(int), cudaMemcpyDeviceToHost);

//     // ====================================================================
//     // HIỂN THỊ MA TRẬN GLCM
//     // ====================================================================
//     // printf("\n--- TONG QUAN MA TRAN GLCM (Kich thuoc thuc: 256x256) ---\n");
//     // printf("In thu 10 hang va 10 cot dau tien de kiem tra:\n");
    
//     // for (int i = 0; i < 10; i++) {
//     //     for (int j = 0; j < 10; j++) {
//     //         // In mỗi số chiếm khoảng trống 5 ký tự để căn lề cho đẹp
//     //         printf("%5d ", h_glcm[i * 256 + j]); 
//     //     }
//     //     printf("\n");
//     // }

//     // Tính tổng để chuẩn hóa
//     double total_pairs = 0;
//     for (int i = 0; i < 256 * 256; i++) total_pairs += h_glcm[i];

//     // Tính Contrast, Homogeneity, Energy... (Đoạn code CPU mình gửi ban nãy)
//     double contrast = 0.0, homogeneity = 0.0, energy = 0.0;
//     for (int i = 0; i < 256; i++) {
//         for (int j = 0; j < 256; j++) {
//             double p = h_glcm[i * 256 + j] / total_pairs;
//             if (p > 0) {
//                 contrast += p * (i - j) * (i - j);
//                 homogeneity += p / (1.0 + abs(i - j));
//                 energy += p * p;
//             }
//         }
//     }

//     printf("=> Contrast    : %f\n", contrast);
//     printf("=> Homogeneity : %f\n", homogeneity);
//     printf("=> Energy      : %f\n", energy);

//     // ====================================================================
//     // BƯỚC CUỐI: LƯU ẢNH KẾT QUẢ ĐỂ KIỂM CHỨNG BẰNG IMAGEJ
//     // ====================================================================
//     // printf("\nDang xuat anh ket qua ra file RAW...\n");
    
//     // 1. Cấp phát mảng tạm trên RAM (CPU) để hứng dữ liệu từ GPU về
//     // unsigned char* h_mip = new unsigned char[frame_pixels];
    
//     // 2. Copy dữ liệu từ Device (GPU) về lại Host (CPU)
//     // cudaMemcpy(h_mip, d_mip, frame_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
//     // 3. Ghi dữ liệu ra các file nhị phân (RAW)
//     // FILE* f_mip = fopen("ketqua_MIP.raw", "wb");
//     // if(f_mip) { fwrite(h_mip, sizeof(unsigned char), frame_pixels, f_mip); fclose(f_mip); }
    
//     // 4. Dọn dẹp RAM máy tính
//     // delete[] h_mip;

//     // printf("Da luu thanh cong file: ketqua_MIP.raw\n");
//     // printf("=> Hay mo file nay bang ImageJ (Size: 1000x2048, 8-bit) de xem ket qua!\n");

//     // 4. DỌN DẸP BỘ NHỚ (Đừng quên giải phóng nhé)
//     cudaFree(d_volume); cudaFree(d_mip); cudaFree(d_glcm);
// }

int main() {
    // 1. THÔNG SỐ CƠ BẢN (Sửa W, H, D theo dataset bạn chọn)
    int W = 1000;
    int H = 2048;
    int D = 500;
    long long total_pixels = (long long)W * H * D;
    long long frame_pixels = (long long)W * H;

    // 2. KHAI BÁO & CẤP PHÁT BỘ NHỚ TRÊN GPU (VRAM)
    unsigned char *d_volume, *d_mip, *d_blurred, *d_binary;
    int *d_white_count;

    cudaMalloc(&d_volume, total_pixels * sizeof(unsigned char));
    cudaMalloc(&d_mip, frame_pixels * sizeof(unsigned char));
    cudaMalloc(&d_blurred, frame_pixels * sizeof(unsigned char));
    cudaMalloc(&d_binary, frame_pixels * sizeof(unsigned char));
    cudaMalloc(&d_white_count, sizeof(int));

    // ====================================================================
    // ĐOẠN ĐỌC DỮ LIỆU TỪ FILE .RAW VÀ ĐẨY LÊN GPU
    // Cần chuyển đổi 500 file ảnh txt thành file volume_3d.raw
    // Nếu không dùng cách này có thể tự thay đổi để đọc dữ liệu
    // ====================================================================
    // Cấp phát mảng tạm trên RAM (CPU)
    unsigned char* h_volume = new unsigned char[total_pixels];

    // Mở và đọc file nhị phân siêu tốc
    printf("Dang doc file volume_3d.raw vao RAM...\n");
    // FILE *f = fopen("volume_3d.raw", "rb");
    FILE *f = fopen("output/o_volume.raw", "rb");
    if (f == NULL) {
        printf("LOI: Khong tim thay file volume_3d.raw. Kiem tra lai thu muc!\n");
        return -1; // Dừng chương trình nếu không có file
    }
    fread(h_volume, sizeof(unsigned char), total_pixels, f);
    fclose(f);

    // Copy dữ liệu từ RAM sang GPU
    printf("Dang day du lieu len GPU...\n");
    cudaMemcpy(d_volume, h_volume, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Giải phóng RAM máy tính (Vì GPU đã giữ bản sao rồi)
    delete[] h_volume;
    // ====================================================================

    // 3. CẤU HÌNH LƯỚI THỢ (Grid & Block)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((W + 15) / 16, (H + 15) / 16);

    // ====================================================================
    // BƯỚC MỞ MÀN: ÉP KHỐI 3D THÀNH ẢNH 2D
    // ====================================================================
    printf("Dang ep khoi 3D xuong 2D (MIP)...\n");
    kernel_MIP_Projection<<<blocksPerGrid, threadsPerBlock>>>(d_volume, d_mip, W, H, D);
    cudaDeviceSynchronize();

    // ====================================================================
    // NHÁNH 1: VESSEL ANALYSIS (PHÂN TÍCH MẠCH MÁU)
    // ====================================================================
    printf("Dang chay Nhanh 1: Phan tich mat do mach mau...\n");
    kernel_Mean_Blur<<<blocksPerGrid, threadsPerBlock>>>(d_mip, d_blurred, W, H);
    cudaDeviceSynchronize();

    kernel_Thresholding<<<blocksPerGrid, threadsPerBlock>>>(d_blurred, d_binary, W, H, 140);
    cudaDeviceSynchronize();

    // Khởi tạo biến đếm bằng 0
    cudaMemset(d_white_count, 0, sizeof(int));
    kernel_Count_Vessels<<<blocksPerGrid, threadsPerBlock>>>(d_binary, d_white_count, W, H);
    cudaDeviceSynchronize();

    int h_white_count = 0;
    cudaMemcpy(&h_white_count, d_white_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("=> Mat do mach mau: %.2f%%\n\n", (float)h_white_count / frame_pixels * 100.0f);

    // ====================================================================
    // BƯỚC CUỐI: LƯU ẢNH KẾT QUẢ ĐỂ KIỂM CHỨNG BẰNG IMAGEJ
    // ====================================================================
    // printf("\nDang xuat anh ket qua ra file RAW...\n");

    // 1. Cấp phát mảng tạm trên RAM (CPU) để hứng dữ liệu từ GPU về
    // unsigned char* h_mip = new unsigned char[frame_pixels];
    // unsigned char* h_blurred = new unsigned char[frame_pixels];
    // unsigned char* h_binary = new unsigned char[frame_pixels];

    // 2. Copy dữ liệu từ Device (GPU) về lại Host (CPU)
    // cudaMemcpy(h_mip, d_mip, frame_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_blurred, d_blurred, frame_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_binary, d_binary, frame_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 3. Ghi dữ liệu ra các file nhị phân (RAW)
    // FILE* f_mip = fopen("ketqua_1_MIP.raw", "wb");
    // if(f_mip) { fwrite(h_mip, sizeof(unsigned char), frame_pixels, f_mip); fclose(f_mip); }

    // FILE* f_blur = fopen("ketqua_2_Blur.raw", "wb");
    // if(f_blur) { fwrite(h_blurred, sizeof(unsigned char), frame_pixels, f_blur); fclose(f_blur); }

    // FILE* f_binary = fopen("ketqua_3_Binary.raw", "wb");
    // if(f_binary) { fwrite(h_binary, sizeof(unsigned char), frame_pixels, f_binary); fclose(f_binary); }

    // 4. Dọn dẹp RAM máy tính
    // delete[] h_mip; delete[] h_blurred; delete[] h_binary;

    // printf("Da luu thanh cong 3 file: ketqua_1_MIP.raw, ketqua_2_Blur.raw, ketqua_3_Binary.raw\n");
    // printf("=> Hay mo 3 file nay bang ImageJ (Size: 1000x2048, 8-bit) de xem ket qua!\n");

    cudaFree(d_volume); cudaFree(d_mip); cudaFree(d_blurred); cudaFree(d_binary); cudaFree(d_white_count);
}
