#include <iostream>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;

// o
const int W = 560;
const int H = 1000;
const int D = 500; 

// ro
// const int W = 560;
// const int H = 1000;
// const int D = 500; // 496 

// normal
// const int W = 1000;
// const int H = 2048;
// const int D = 500; 

int main()
{
    const long long pixels_per_slice = (long long)W * H;

    FILE* out =
        fopen("volume_3d_ro.raw", "wb");

    // FILE* out =
    //     fopen("volume_3d_o.raw", "wb");

    // FILE* out =
    //     fopen("volume_3d_normal.raw", "wb");

    if (!out) {
        cerr << "Khong tao duoc file output\n";
        return -1;
    }

    vector<float> slice(pixels_per_slice);

    int valid_slices = 0;
    int missing_files = 0;
    int corrupted_files = 0;

    cout << "Bat dau tao volume..." << endl;

    for (int z = 1; z <= D; z++) {
        string filename = "output/ro/ro_" + to_string(z) + ".txt";
        // string filename = "output/o/o_" + to_string(z) + ".txt";
        // string filename = "output/normal/" + to_string(z) + ".txt";

        FILE* in = fopen(filename.c_str(), "r");

        if (!in) {
            cerr << "[SKIP] Missing: " << filename << endl;
            missing_files++;
            continue;
        }

        bool ok = true;

        for (long long i = 0; i < pixels_per_slice; i++) {
            if (fscanf(in, "%f", &slice[i]) != 1) {
                ok = false;
                break;
            }
        }

        fclose(in);

        if (!ok) {
            cerr << "[SKIP] Corrupted: " << filename << endl;
            corrupted_files++;
            continue;
        }

        fwrite(slice.data(), sizeof(float), pixels_per_slice, out);
        valid_slices++;

        cout << "\rLoaded " << valid_slices << " slices" << flush;
    }

    fclose(out);

    cout << "\n\n========== SUMMARY ==========\n";

    cout << "Valid slices     : " << valid_slices << endl;
    cout << "Missing files    : " << missing_files << endl;
    cout << "Corrupted files  : " << corrupted_files << endl;

    cout << "Volume size      : " << W << " x " << H << " x " << valid_slices << endl;

    cout << "RAW created: volume_3d_ro.raw" << endl;
    // cout << "RAW created: volume_3d_o.raw" << endl;
    // cout << "RAW created: volume_3d_normal.raw" << endl;

    return 0;
}