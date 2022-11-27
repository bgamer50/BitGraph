#pragma once

#include <cstdint>
#include <boost/any.hpp>
#include <boost/core/typeinfo.hpp>
#include <sstream>

#include <unordered_map>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "traversal/Comparison.h"

namespace bitgraph {
    namespace memory {
        /*
            desired_elements : The ids of the elements to search for.
            element_list : The list of existing elements to search through.
            search_pointers : Will be populated with the index of found elements (if == N_existing it was not found), starts as ptr for binary search.
            N_desired : The number of elements being searched for.
            N_existing : The number of elements in the element list.
        */
        __global__ void k_search_index(uint64_t* desired_elements, uint64_t* element_list, size_t* search_pointers, size_t N_desired, size_t N_existing) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            size_t R = 1 + log2f(N_existing);

            for(size_t k = index; k < N_desired; k += stride) {
                // runs R - 1 times
                for(size_t i = 1; i < R; ++i) {
                    // look for the index
                    size_t ptr = search_pointers[k];
                    int diff = desired_elements[k] - element_list[ptr];
                    if(diff == 0) search_pointers[k] = ptr;
                    else search_pointers[k] = (diff < 0) ? (ptr / 2) : (ptr + N_existing) / 2;
                }

                // Final check (Rth time)
                if(desired_elements[k] - element_list[search_pointers[k]] != 0) search_pointers[k] = N_existing;
            }
        }

        struct not_found {
            size_t N=0;
            __host__ __device__ bool operator()(const size_t x) {
                return x >= N;
            }
        };
        
        /*
            Returns (found elements, num found elements)
        */
        std::tuple<size_t*, size_t> search_index(uint64_t* desired_elements, uint64_t* element_list, size_t N_desired, const size_t N_existing) {
            size_t* found_elements;
            cudaMalloc(&found_elements, sizeof(size_t) * N_desired);
            cudaMemset(found_elements, N_existing / 2, N_desired);

            size_t block_size = 128;
            size_t num_blocks = N_desired / block_size;

            k_search_index<<<block_size, num_blocks>>>(desired_elements, element_list, found_elements, N_desired, N_existing);
            cudaDeviceSynchronize();

            auto found_elements_tptr = thrust::device_pointer_cast(found_elements);
            auto nf = not_found();
            nf.N = N_existing;
            auto N_found_ptr = thrust::remove_if(found_elements_tptr, found_elements_tptr + N_desired, nf);
            size_t N_found = N_found_ptr - found_elements_tptr;

            return std::make_tuple(found_elements, N_found);
        }

        void copy(std::vector<boost::any>& dst_host, void* values, size_t* select, size_t N_select, gremlinxx::comparison::C dtype) {
            auto select_tptr = thrust::device_pointer_cast(select);
            switch(dtype) {
                case gremlinxx::comparison::C::UINT64: {
                    std::vector<uint64_t> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((uint64_t*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
                case gremlinxx::comparison::C::UINT32: {
                    std::vector<uint32_t> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((uint32_t*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
                case gremlinxx::comparison::C::UINT8: {
                    std::vector<uint8_t> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((uint8_t*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
                case gremlinxx::comparison::C::INT64: {
                    std::vector<int64_t> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((int64_t*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
                case gremlinxx::comparison::C::INT32: {
                    std::vector<int32_t> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((int32_t*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
                case gremlinxx::comparison::C::INT8: {
                    std::vector<int8_t> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((int8_t*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
                case gremlinxx::comparison::C::FLOAT64: {
                    std::vector<double> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((double*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
                case gremlinxx::comparison::C::FLOAT32: {
                    std::vector<float> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((float*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    break;
                }
                case gremlinxx::comparison::C::STRING: {
                    std::vector<uint64_t> vec(N_select);
                    auto values_tptr = thrust::device_pointer_cast((uint64_t*)values);
                    thrust::copy(
                        thrust::make_permutation_iterator(values_tptr, select_tptr),
                        thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                        vec.begin()
                    );
                    dst_host.insert(dst_host.begin(), vec.begin(), vec.end());
                    break;
                }
            }
        }

        class GPUPropertyIndex {
            private:
                // Element, value pairs
                uint64_t* elements;
                void* values;
                size_t size;
                size_t max_size;

                gremlinxx::comparison::C dtype;
                size_t value_size;

                bool sorted = false;

            public:
                GPUPropertyIndex(size_t initial_max_size, gremlinxx::comparison::C dtype){
                    this->dtype = dtype;
                    this->value_size = gremlinxx::comparison::C_size[dtype];
                    this->max_size = initial_max_size;
                    this->size = 0;

                    cudaMallocManaged(&this->elements, sizeof(uint64_t) * this->max_size);
                    cudaMemAdvise(this->elements, sizeof(uint64_t) * this->max_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

                    cudaMallocManaged(&this->values, this->value_size * this->max_size);
                    cudaMemAdvise(this->elements, this->value_size * this->max_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                }

                void sort() {
                    auto elements_tptr = thrust::device_pointer_cast(elements);
                    switch(this->dtype) {
                        case gremlinxx::comparison::C::UINT64: {
                            auto values_tptr = thrust::device_pointer_cast((uint64_t*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::UINT32: {
                            auto values_tptr = thrust::device_pointer_cast((uint32_t*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::UINT8: {
                            auto values_tptr = thrust::device_pointer_cast((uint8_t*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::INT64: {
                            auto values_tptr = thrust::device_pointer_cast((int64_t*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::INT32: {
                            auto values_tptr = thrust::device_pointer_cast((int32_t*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::INT8: {
                            auto values_tptr = thrust::device_pointer_cast((int8_t*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::FLOAT64: {
                            auto values_tptr = thrust::device_pointer_cast((double*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::FLOAT32: {
                            auto values_tptr = thrust::device_pointer_cast((float*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                        case gremlinxx::comparison::C::STRING: {
                            auto values_tptr = thrust::device_pointer_cast((uint64_t*)values);
                            thrust::sort_by_key(elements_tptr, elements_tptr + size, values_tptr);
                            break;
                        }
                    }

                    this->sorted = true;
                }

                void resize() {
                    this->max_size *= 1.3;

                    uint64_t* new_elements;
                    cudaMallocManaged(&new_elements, sizeof(uint64_t) * this->max_size);
                    cudaMemcpy(new_elements, this->elements, sizeof(uint64_t) * this->size, cudaMemcpyDefault);
                    cudaFree(this->elements);
                    this->elements = new_elements;
                    
                    void* new_values;
                    cudaMallocManaged(&new_values, this->value_size * this->max_size);
                    cudaMemcpy(new_values, this->values, this->value_size * this->size, cudaMemcpyDefault);
                    cudaFree(this->values);
                    this->values = new_values;
                }

                std::vector<boost::any> get(std::vector<uint64_t> desired_element_ids) {
                    size_t* desired_element_ids_device;
                    size_t N_desired = desired_element_ids.size();
                    cudaMalloc(&desired_element_ids_device, sizeof(uint64_t) * N_desired);
                    cudaMemcpy(desired_element_ids_device, desired_element_ids.data(), N_desired, cudaMemcpyHostToDevice);

                    if(!this->sorted) this->sort();

                    size_t* found_elements;
                    size_t N_found;
                    std::tie(found_elements, N_found) = search_index(desired_element_ids_device, this->elements, desired_element_ids.size(), this->size);
                    cudaFree(desired_element_ids_device);

                    std::vector<boost::any> found_elements_host;
                    copy(found_elements_host, values, found_elements, N_found, this->dtype);

                    cudaFree(found_elements);
                    return found_elements_host;
                }

                void set(uint64_t element_id, boost::any value) {
                    if(this->size == this->max_size) resize();

                    this->elements[size] = element_id;
                    try {
                        switch(this->dtype) {
                            case gremlinxx::comparison::C::UINT64: {
                                ((uint64_t*)this->values)[size] = boost::any_cast<uint64_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::UINT32: {
                                ((uint32_t*)this->values)[size] = boost::any_cast<uint32_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::UINT8: {
                                ((uint8_t*)this->values)[size] = boost::any_cast<uint8_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::INT64: {
                                ((int64_t*)this->values)[size] = boost::any_cast<int64_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::INT32: {
                                ((int32_t*)this->values)[size] = boost::any_cast<int32_t>(value);  
                                break;
                            }
                            case gremlinxx::comparison::C::INT8: {
                                ((int8_t*)this->values)[size] = boost::any_cast<int8_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::FLOAT64: {
                                ((double*)this->values)[size] = boost::any_cast<double>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::FLOAT32: {
                                ((float*)this->values)[size] = boost::any_cast<float>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::STRING: {
                                ((uint64_t*)this->values)[size] = boost::any_cast<uint64_t>(value);
                                break;
                            }
                        }
                    }
                    catch(const boost::bad_any_cast& ex) {
                        std::stringstream sx;
                        sx << ex.what() << std::endl;
                        sx << "expected type: " << gremlinxx::comparison::C_to_string[this->dtype] << std::endl;
                        sx << "but got:" << std::endl;
                        sx << boost::core::demangled_name(value.type()) << std::endl;
                        throw std::runtime_error(sx.str());
                    }

                    ++size;
                    this->sorted = false;
                }
        };
        
        class GPUPropertyTable {
            private:
                std::unordered_map<std::string, GPUPropertyIndex> key_to_index_map;
            public:
                GPUPropertyTable(){}
        };
    }
}