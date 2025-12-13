#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <chrono>

int main() {
    int N = 100000;
    std::vector<int> v(N);
    for (int i = 0; i < N; i++) {
        v[i] = rand() % 100000;
    }
    std::vector<int> prefix_sum(N);
    auto start = std::chrono::high_resolution_clock::now();
    prefix_sum[0] = v[0];
    for (int i = 1; i < N; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + v[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    std::cout << "Prefix sum: " << prefix_sum[N - 1] << std::endl;
    return 0;
}