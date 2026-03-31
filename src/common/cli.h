#pragma once

#include "src/common/cuda_check.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class CliArgs {
public:
    CliArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string token = argv[i];
            if (token.rfind("--", 0) != 0) {
                positional_.push_back(token);
                continue;
            }
            token = token.substr(2);
            std::string key = token;
            std::string value = "1";
            const auto eq_pos = token.find('=');
            if (eq_pos != std::string::npos) {
                key = token.substr(0, eq_pos);
                value = token.substr(eq_pos + 1);
            } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                value = argv[++i];
            }
            options_[key] = value;
        }
    }

    bool has(const std::string& key) const {
        return options_.find(key) != options_.end();
    }

    std::string get_string(const std::string& key, const std::string& fallback) const {
        const auto it = options_.find(key);
        return it == options_.end() ? fallback : it->second;
    }

    int get_int(const std::string& key, int fallback) const {
        const auto it = options_.find(key);
        return it == options_.end() ? fallback : std::stoi(it->second);
    }

    float get_float(const std::string& key, float fallback) const {
        const auto it = options_.find(key);
        return it == options_.end() ? fallback : std::stof(it->second);
    }

    bool get_bool(const std::string& key, bool fallback) const {
        const auto it = options_.find(key);
        if (it == options_.end()) {
            return fallback;
        }
        const std::string value = lower(it->second);
        return value == "1" || value == "true" || value == "yes" || value == "on";
    }

    static std::string lower(std::string text) {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return text;
    }

private:
    std::unordered_map<std::string, std::string> options_;
    std::vector<std::string> positional_;
};

inline void print_device_summary() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::cout << "CUDA devices: " << device_count << std::endl;
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop {};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "  [" << device << "] " << prop.name << " sm_" << prop.major << prop.minor
                  << " global_mem=" << (prop.totalGlobalMem / (1024 * 1024)) << " MiB" << std::endl;
    }
}
