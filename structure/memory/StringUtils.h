#pragma once

#include <string>
#include <functional>

namespace bitgraph {
    namespace memory {
        class StringIndex {
            private:
                // Possible TODO: reference counting
                std::unordered_map<size_t, std::string> reverse_lookup_table;

            public:
                inline size_t from_cpu_value(std::string s) {
                    size_t v = std::hash<std::string>{}(s);
                    if(this->reverse_lookup_table.find(v) == this->reverse_lookup_table.end()) {
                        this->reverse_lookup_table[v] = s;
                    }

                    return v;
                }

                inline std::string from_gpu_value(size_t v) {
                    return this->reverse_lookup_table[v];
                }

                inline std::vector<boost::any> from_cpu_anys(std::vector<boost::any> cpu_anys) {
                    std::vector<boost::any> gpu_anys;
                    gpu_anys.reserve(cpu_anys.size());

                    for(boost::any& b : cpu_anys) {
                        gpu_anys.push_back(from_cpu_value(boost::any_cast<std::string>(b)));
                    }

                    return gpu_anys;
                }

                inline std::vector<boost::any> from_gpu_anys(std::vector<boost::any> gpu_anys) {
                    std::vector<boost::any> cpu_anys;
                    cpu_anys.reserve(gpu_anys.size());

                    for(boost::any& b : gpu_anys) {
                        cpu_anys.push_back(from_gpu_value(boost::any_cast<uint64_t>(b)));
                    }

                    return cpu_anys;
                }
        };
    }
}