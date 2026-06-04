// Code dung de chuyen 500 file anh txt thanh file raw
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

// �?t k�ch thu?c theo d�ng file c?a b?n
const int W = 1000; 
const int H = 2048; 
const int D = 20;  

int main() {
    long long total_pixels = (long long)W * H * D;
    
    // T?o m?ng 8-bit (unsigned char) d? ImageJ d? d?c nh?t
    unsigned char* raw_data = new unsigned char[total_pixels];

    cout << "Bat dau doc 500 file txt. Vui long doi..." << endl;

    for (int z = 0; z < D; z++) {
        string filename = "output/o/o_" + to_string(z + 1) + ".txt";
        
        // D�ng fopen thay v� ifstream d? d?c si�u t?c
        FILE* file = fopen(filename.c_str(), "r");
        if (!file) {
            cout << "Loi: Khong mo duoc file " << filename << endl;
            return -1;
        }

        long long offset = (long long)z * W * H;
        float val;
        
        // �?c t?ng con s? trong file txt
        for (int i = 0; i < W * H; i++) {
            if (fscanf(file, "%f", &val) != 1) break;

            // B? ph�p chia 4095 di! Gi? nguy�n gi� tr? g?c.
            int pixel = (int)val;

            raw_data[offset + i] = (unsigned char)pixel;
        }
        fclose(file);

        if ((z + 1) % 50 == 0) cout << "Da doc xong " << z + 1 << "/500 file..." << endl;
    }

    cout << "------------------------------------------" << endl;
    cout << "Dang ghi toan bo ra file volume_3d.raw..." << endl;
    
    // Ghi m?ng RAM ra 1 file nh? ph�n duy nh?t
    // FILE* out_file = fopen("volume_3d.raw", "wb");
    FILE* out_file = fopen("volume_3d_o.raw", "wb");
    fwrite(raw_data, sizeof(unsigned char), total_pixels, out_file);
    fclose(out_file);

    delete[] raw_data;
    
    cout << "XUAT FILE THANH CONG! Bay gio ban co the dung file volume_3d.raw cho code CUDA" << endl;
    return 0;
}
