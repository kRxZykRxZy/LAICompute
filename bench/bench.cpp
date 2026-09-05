#include "laic/matmul.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr size_t N = 512;
    std::vector<float> A(N*N), B(N*N), C(N*N);
    for (size_t i=0;i<A.size();++i) { A[i]=std::sin(float(i)*0.001f); B[i]=std::cos(float(i)*0.001f); }
    laic::MatmulConfig cfg;
    const auto t0 = std::chrono::steady_clock::now();
    laic::matmul_tiled(A.data(), B.data(), C.data(), N, N, N, cfg);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1-t0).count();
    const double gflops = (2.0*N*N*N) / sec / 1e9;
    double checksum=0; for (float x:C) checksum += x;
    std::cout << "512x512 tiled matmul\n" << "time: " << sec << " s\n" << "GFLOP/s: " << gflops << "\n" << "checksum: " << checksum << "\n";
}
