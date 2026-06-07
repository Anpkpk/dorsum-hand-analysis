# Dorsum Hand Analysis using CUDA GPU

ỨNG DỤNG LẬP TRÌNH SONG SONG CUDA C++ TRONG XỬ LÝ VÀ PHÂN
TÍCH KHỐI DỮ LIỆU ẢNH Y KHOA 3D
---

## Tổng quan hệ thống

Dự án được thiết kế nhằm tận dụng khả năng tính toán song song của GPU NVIDIA để tăng tốc quá trình xử lý ảnh y sinh học. Quy trình gồm hai giai đoạn chính:

1. Tiền xử lý dữ liệu trên CPU.
2. Phân tích và trích xuất đặc trưng trên GPU.

### Workflow

```text
TXT Slices
    │
    ▼
Convert to Binary
    │
    ▼
3D Volume (.raw)
    │
    ▼
Upload to GPU (VRAM)
    │
    ▼
En-face MIP Projection
    │
    ▼
2D Projection Image
    │
 ┌──┴─────────────────────┐
 ▼                        ▼

Vessel Analysis      Texture Analysis
 ▼                        ▼

Mean Blur            GLCM Construction
 ▼                        ▼

Thresholding         Feature Extraction
 ▼                        ▼

Vessel Density       Contrast
                      Energy
                      Homogeneity
```

---

## Chức năng chính

### 1. Tiền xử lý dữ liệu

* Đọc các lát cắt ảnh từ file văn bản (`.txt`)
* Chuyển đổi dữ liệu thành thể tích 3D
* Lưu dưới dạng file nhị phân (`.raw`)

### 2. En-face Maximum Intensity Projection (MIP)

* Chiếu thể tích 3D thành ảnh 2D
* Giữ lại giá trị cường độ lớn nhất theo chiều sâu
* Làm nổi bật cấu trúc mạch máu bề mặt

### 3. Phân tích mạch máu

Các bước xử lý:

1. Mean Blur
2. Thresholding
3. Đếm pixel mạch máu

Kết quả:

* Số lượng pixel mạch máu
* Mật độ mạch máu (%)

### 4. Phân tích kết cấu mô

Sử dụng Gray-Level Co-occurrence Matrix (GLCM) để tính:

* Contrast
* Energy
* Homogeneity

---

## Cấu trúc mã nguồn

```text
.
├── img.cpp
├── kernel.h
├── main.cu
└── README.md
```

### img.cpp

Tiền xử lý dữ liệu trên CPU:

* Đọc các file TXT
* Ghép lát cắt
* Tạo volume 3D
* Xuất file `.raw`

### kernel.h

Chứa các CUDA kernels:

```cpp
kernel_Enface_MIP()
kernel_Mean_Blur()
kernel_Thresholding()
kernel_Count_Vessels()
kernel_Compute_GLCM()
```

### main.cu

Chương trình điều khiển chính:

* Đọc volume
* Cấp phát bộ nhớ GPU
* Cấu hình Grid/Block
* Gọi CUDA kernels
* Ghi kết quả ra file

---

## Yêu cầu hệ thống

### Phần cứng

* NVIDIA GPU hỗ trợ CUDA

### Phần mềm

* CUDA Toolkit 11.0+
* GCC/G++ hoặc MSVC
* C++11 trở lên
---

## Cấu trúc dữ liệu đầu vào

```text
output/
├── ro/
│   ├── ro_1.txt
│   ├── ro_2.txt
│   └── ...
├── o/
│   ├── o_1.txt
│   ├── o_2.txt
│   └── ...
└── normal/
    ├── 1.txt
    ├── 2.txt
    └── ...
```

---

# Hướng dẫn biên dịch và chạy

## Bước 1: Tạo volume 3D

Biên dịch:

```bash
g++ -03 img.cpp -o generate_volume
```

Chạy:

```bash
./generate_volume
```

---

## Bước 2: Chạy chương trình CUDA

Biên dịch:

```bash
nvcc -O3 main.cu -o hand_analysis
```

Chạy:

```bash
./hand_analysis
```

---

### Các file ảnh kết quả

```text
ketqua_1_Enface.raw
ketqua_2_Blur.raw
ketqua_3_Binary.raw
```

#### ketqua_1_Enface.raw

Ảnh En-face MIP sau khi chiếu từ thể tích 3D.

#### ketqua_2_Blur.raw

Ảnh sau bước lọc nhiễu bằng Mean Blur.

#### ketqua_3_Binary.raw

Ảnh nhị phân phục vụ phân tích mật độ mạch máu.

---

## Cấu hình thông số

Các tham số có thể thay đổi trong:

```text
img.cpp
main.cu
```

### Kích thước volume

Với o và ro:
```cpp
W = 560;
H = 1000;
D = 500;
```

Trong đó:

| Tham số | Ý nghĩa           |
| ------- | ----------------- |
| W       | Chiều rộng        |
| H       | Chiều cao         |
| D       | Chiều sâu         |

### Ngưỡng phân đoạn mạch máu

Trong:

```cpp
kernel_Thresholding(...)
```

Giá trị mặc định:

```cpp
45.0f
```

Có thể điều chỉnh để phù hợp với từng bộ dữ liệu.

---

## Đặc điểm nổi bật

* Tăng tốc bằng CUDA GPU
* Xử lý dữ liệu thể tích 3D
* En-face MIP thời gian thực
* Phân tích mật độ mạch máu tự động
* Trích xuất đặc trưng kết cấu bằng GLCM
* Dễ dàng mở rộng cho các nghiên cứu ảnh y sinh học và OCT/OCTA

---
