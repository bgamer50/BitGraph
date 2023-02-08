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
                size_t from_cpu_value(std::string s) {
                    size_t v = std::hash<std::string>{}(s);
                    if(this->reverse_lookup_table.find(v) == this->reverse_lookup_table.end()) {
                        this->reverse_lookup_table[v] = s;
                    }

                    return v;
                }

                std::string from_gpu_value(size_t v) {
                    return this->reverse_lookup_table[v];
                }
        };
    }
}