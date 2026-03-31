#pragma once

#include "src/common/cli.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

enum class DataType {
    kFloat32,
    kFloat16,
};

inline DataType parse_dtype(const std::string& text) {
    const std::string lowered = CliArgs::lower(text);
    if (lowered == "fp16" || lowered == "half" || lowered == "float16") {
        return DataType::kFloat16;
    }
    return DataType::kFloat32;
}

inline std::string dtype_name(DataType dtype) {
    return dtype == DataType::kFloat16 ? "fp16" : "fp32";
}

template <typename T>
__host__ __device__ inline float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__host__ __device__ inline float to_float<half>(half value) {
    return __half2float(value);
}

template <typename T>
__host__ __device__ inline T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ inline half from_float<half>(float value) {
    return __float2half(value);
}

template <typename T>
inline std::vector<T> random_vector(std::size_t size, float scale = 1.0f, int seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<T> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = from_float<T>(dist(rng));
    }
    return values;
}

template <typename T>
inline std::vector<T> zeros_vector(std::size_t size) {
    return std::vector<T>(size, from_float<T>(0.0f));
}

template <typename T>
inline bool compare_vectors(
    const std::vector<T>& lhs,
    const std::vector<T>& rhs,
    float atol,
    float rtol,
    float* max_abs_err,
    float* max_rel_err
) {
    *max_abs_err = 0.0f;
    *max_rel_err = 0.0f;
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const float a = to_float(lhs[i]);
        const float b = to_float(rhs[i]);
        const float abs_err = std::fabs(a - b);
        const float rel_err = abs_err / std::max(std::fabs(b), 1e-6f);
        *max_abs_err = std::max(*max_abs_err, abs_err);
        *max_rel_err = std::max(*max_rel_err, rel_err);
        if (abs_err > atol && rel_err > rtol) {
            return false;
        }
    }
    return true;
}

inline void print_csv_row(const std::vector<std::pair<std::string, std::string>>& fields) {
    for (std::size_t i = 0; i < fields.size(); ++i) {
        std::cout << fields[i].second;
        if (i + 1 != fields.size()) {
            std::cout << ",";
        }
    }
    std::cout << std::endl;
}
