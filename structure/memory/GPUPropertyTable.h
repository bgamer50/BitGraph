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

#include "structure/memory/StringUtils.h"

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
            cudaDeviceSynchronize();
            cudaCheckErrors("allocate found elements");

            size_t block_size = 128;
            size_t num_blocks = N_desired / block_size;

            k_search_index<<<block_size, num_blocks>>>(desired_elements, element_list, found_elements, N_desired, N_existing);
            cudaDeviceSynchronize();
            cudaCheckErrors("k_search_index");

            auto found_elements_tptr = thrust::device_pointer_cast(found_elements);
            auto nf = not_found();
            nf.N = N_existing;
            auto N_found_ptr = thrust::remove_if(found_elements_tptr, found_elements_tptr + N_desired, nf);
            size_t N_found = N_found_ptr - found_elements_tptr;

            return std::make_tuple(found_elements, N_found);
        }

        struct found_op : public thrust::unary_function<size_t, bool> {
            size_t N = 0;

            __host__ __device__ bool operator()(const size_t x) const {
                return x < N;
            }

        };

        /*
            Returns (found elements, original indices, num found elements)
            Same as search_index, but also returns the original indices (also of size = # found)
        */
        std::tuple<size_t*, size_t*, size_t> search_index_device(uint64_t* desired_elements, uint64_t* element_list, size_t N_desired, const size_t N_existing) {
            size_t* found_elements;
            cudaMalloc(&found_elements, sizeof(size_t) * N_desired);
            cudaMemset(found_elements, N_existing / 2, N_desired);
            cudaDeviceSynchronize();
            cudaCheckErrors("Allocate found elements");

            size_t block_size = 128;
            size_t num_blocks = N_desired / block_size;

            k_search_index<<<block_size, num_blocks>>>(desired_elements, element_list, found_elements, N_desired, N_existing);
            cudaDeviceSynchronize();
            cudaCheckErrors("k_search_index");

            auto found_elements_tptr = thrust::device_pointer_cast(found_elements);
            auto fo = found_op();
            fo.N = N_existing;

            size_t* found_indices;
            cudaMalloc(&found_indices, sizeof(size_t) * N_desired);
            cudaDeviceSynchronize();
            cudaCheckErrors("allocate found_indices");

            auto found_indices_tptr = thrust::device_pointer_cast(found_indices);
            auto found_indices_end = thrust::copy_if(
                thrust::make_counting_iterator((size_t)0),
                thrust::make_counting_iterator((size_t)N_desired),
                found_elements_tptr,
                found_indices_tptr,
                fo
            );
            size_t N_found = found_indices_end - found_indices_tptr;

            // FIXME this may be wasting memory.
            size_t* new_found_elements;
            cudaMalloc(&new_found_elements, sizeof(size_t) * N_found);
            cudaDeviceSynchronize();
            cudaCheckErrors("allocate new found elements");

            thrust::device_ptr<size_t> new_found_elements_tptr = thrust::device_pointer_cast(new_found_elements);
            thrust::copy(
                thrust::make_permutation_iterator(found_elements_tptr, found_indices_tptr),
                thrust::make_permutation_iterator(found_elements_tptr, found_indices_tptr + N_found),
                new_found_elements_tptr
            );
            cudaFree(found_elements);

            return std::make_tuple(new_found_elements, found_indices, N_found);
        }

        void copy_found_values(std::vector<boost::any>& dst_host, void* values, size_t* select, size_t N_select, gremlinxx::comparison::C dtype) {
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

            cudaDeviceSynchronize();
            cudaCheckErrors("copy values");
        }

        /*
            Copy from values to a device array.

        */
        template<typename T>
        void copy_found_values_device_helper(void* out, void* values, size_t* select, size_t N_select) {
            auto select_tptr = thrust::device_pointer_cast(select);
            auto out_tptr = thrust::device_pointer_cast((T*)out);
            auto values_tptr = thrust::device_pointer_cast((T*)values);
            thrust::copy(
                thrust::make_permutation_iterator(values_tptr, select_tptr),
                thrust::make_permutation_iterator(values_tptr, select_tptr + N_select),
                out_tptr
            );
        }

        template
        void copy_found_values_device_helper<uint64_t>(void* out, void* values, size_t* select, size_t N_select);
        template
        void copy_found_values_device_helper<uint32_t>(void* out, void* values, size_t* select, size_t N_select);
        template
        void copy_found_values_device_helper<uint8_t>(void* out, void* values, size_t* select, size_t N_select);
        template
        void copy_found_values_device_helper<int64_t>(void* out, void* values, size_t* select, size_t N_select);
        template
        void copy_found_values_device_helper<int32_t>(void* out, void* values, size_t* select, size_t N_select);
        template
        void copy_found_values_device_helper<int8_t>(void* out, void* values, size_t* select, size_t N_select);
        template
        void copy_found_values_device_helper<double>(void* out, void* values, size_t* select, size_t N_select);
        template
        void copy_found_values_device_helper<float>(void* out, void* values, size_t* select, size_t N_select);

        void copy_found_values_device(void* out, void* values, size_t* select, size_t N_select, gremlinxx::comparison::C dtype) {
            switch(dtype) {
                case gremlinxx::comparison::C::UINT64: {
                    return copy_found_values_device_helper<uint64_t>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::UINT32: {
                    return copy_found_values_device_helper<uint32_t>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::UINT8: {
                    return copy_found_values_device_helper<uint8_t>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::INT64: {
                    return copy_found_values_device_helper<int64_t>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::INT32: {
                    return copy_found_values_device_helper<int32_t>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::INT8: {
                    return copy_found_values_device_helper<int8_t>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::FLOAT64: {
                    return copy_found_values_device_helper<double>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::FLOAT32: {
                    return copy_found_values_device_helper<float>(out, values, select, N_select);
                }
                case gremlinxx::comparison::C::STRING: {
                    return copy_found_values_device_helper<uint64_t>(out, values, select, N_select);
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

                // String info
                bitgraph::memory::StringIndex string_index;

            public:
                GPUPropertyIndex(size_t initial_max_size, gremlinxx::comparison::C dtype){
                    this->dtype = dtype;
                    this->value_size = gremlinxx::comparison::C_size[dtype];
                    this->max_size = initial_max_size;
                    this->size = 0;

                    cudaMallocManaged(&this->elements, sizeof(uint64_t) * this->max_size);
                    //cudaMemAdvise(this->elements, sizeof(uint64_t) * this->max_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("allocate elements");

                    cudaMallocManaged(&this->values, this->value_size * this->max_size);
                    //cudaMemAdvise(this->values, this->value_size * this->max_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("allocate values");
                }

                GPUPropertyIndex() {
                    GPUPropertyIndex(0, gremlinxx::comparison::C::INT64);
                }

                inline gremlinxx::comparison::C get_dtype() {
                    return this->dtype;
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
                    cudaDeviceSynchronize();
                    cudaCheckErrors("allocate new elements");

                    cudaFree(this->elements);
                    this->elements = new_elements;
                    
                    void* new_values;
                    cudaMallocManaged(&new_values, this->value_size * this->max_size);
                    cudaMemcpy(new_values, this->values, this->value_size * this->size, cudaMemcpyDefault);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("allocate new values");

                    cudaFree(this->values);
                    this->values = new_values;
                }

                std::vector<boost::any> get(std::vector<uint64_t>& desired_element_ids) {
                    size_t* desired_element_ids_device;
                    size_t N_desired = desired_element_ids.size();
                    cudaMalloc(&desired_element_ids_device, sizeof(uint64_t) * N_desired);
                    cudaMemcpy(desired_element_ids_device, desired_element_ids.data(), N_desired, cudaMemcpyDefault);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("allocate desired element ids");

                    if(!this->sorted) this->sort();

                    size_t* found_elements;
                    size_t N_found;
                    std::tie(found_elements, N_found) = search_index(desired_element_ids_device, this->elements, desired_element_ids.size(), this->size);
                    cudaFree(desired_element_ids_device);

                    std::vector<boost::any> found_elements_host;
                    copy_found_values(found_elements_host, this->values, found_elements, N_found, this->dtype);

                    cudaFree(found_elements);
                    if(this->dtype == gremlinxx::comparison::C::STRING) {
                        for(auto it = found_elements_host.begin(); it != found_elements_host.end(); ++it) {
                            *it = this->string_index.from_gpu_value(boost::any_cast<size_t>(*it));
                        }
                    }
                    return found_elements_host;
                }

                /*
                    Returns (original indices, values, # found)
                */
                std::tuple<size_t*, void*, size_t> get_device(uint64_t* desired_element_ids, size_t N_desired) {
                    if(!this->sorted) this->sort();

                    size_t* found_elements;
                    size_t* found_indices;
                    size_t N_found;

                    std::tie(found_elements, found_indices, N_found) = search_index_device(
                        desired_element_ids, // these are already on device
                        this->elements,
                        N_desired,
                        this->size
                    );

                    void* found_elements_device;
                    cudaMalloc(&found_elements_device, this->value_size * N_found);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("allocate found elements");
                    copy_found_values_device(
                        found_elements_device,
                        this->values,
                        found_elements,
                        N_found,
                        this->dtype
                    );

                    cudaFree(found_elements);
                    return std::make_tuple(
                        found_indices,
                        found_elements_device,
                        N_found
                    );
                }

                void set(uint64_t element_id, boost::any value) {
                    if(this->size == this->max_size) resize();

                    this->elements[size] = element_id;
                    try {
                        switch(this->dtype) {
                            case gremlinxx::comparison::C::UINT64: {
                                
                                static_cast<uint64_t*>(this->values)[size] = boost::any_cast<uint64_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::UINT32: {
                                static_cast<uint32_t*>(this->values)[size] = boost::any_cast<uint32_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::UINT8: {
                                static_cast<uint8_t*>(this->values)[size] = boost::any_cast<uint8_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::INT64: {
                                static_cast<int64_t*>(this->values)[size] = boost::any_cast<int64_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::INT32: {
                                static_cast<int32_t*>(this->values)[size] = boost::any_cast<int32_t>(value);  
                                break;
                            }
                            case gremlinxx::comparison::C::INT8: {
                                static_cast<int8_t*>(this->values)[size] = boost::any_cast<int8_t>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::FLOAT64: {
                                static_cast<double*>(this->values)[size] = boost::any_cast<double>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::FLOAT32: {
                                static_cast<float*>(this->values)[size] = boost::any_cast<float>(value);
                                break;
                            }
                            case gremlinxx::comparison::C::STRING: {
                                size_t store_value = this->string_index.from_cpu_value(
                                    boost::any_cast<std::string>(value)
                                );

                                static_cast<uint64_t*>(this->values)[size] = store_value;
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
        
        /*
            Wraps an unordered map of property index objects.
        */
        class GPUPropertyTable {
            private:
                std::unordered_map<std::string, GPUPropertyIndex> key_to_index_map;
            public:
                GPUPropertyTable(){}

                void declare_property(std::string property_name, gremlinxx::comparison::C property_dtype, size_t initial_max_size) {
                    this->key_to_index_map[property_name] = GPUPropertyIndex(
                        initial_max_size,
                        property_dtype
                    );
                }

                void set_property_value(uint64_t element, std::string property_name,  boost::any property_value) {
                    this->key_to_index_map[property_name].set(element, property_value);
                }

                boost::any get_property_value(uint64_t element, std::string property_name) {
                    std::vector<uint64_t> vec;
                    vec.push_back(element);
                    std::vector<boost::any> found_values = this->key_to_index_map[property_name].get(vec);
                    if(found_values.empty()) {
                        std::stringstream sx;
                        sx << "No value was found for property " << property_name << " on vertex " << element;
                        throw std::runtime_error(sx.str());
                    }

                    return found_values.front();
                }

                /*
                    Gets the values for the given property on the given elements.
                    If strict = true (default) then an exception will be thrown if one or more elements are missing the property.
                    If strict = false then however many values were found will be returned.
                */
                std::vector<boost::any> get_property_values(std::vector<uint64_t> elements, std::string property_name, bool strict=true) {
                    if(!has_property_key(property_name)) {
                        std::stringstream sx;
                        sx << "Invalid property key " << property_name;
                        throw std::runtime_error(sx.str());
                    }

                    auto found_values = this->key_to_index_map[property_name].get(elements);
                    if(strict && found_values.size() < elements.size()) {
                        std::stringstream sx;
                        sx << "Some elements were missing a value for the property " << property_name;
                        throw std::runtime_error(sx.str());
                    }

                    return found_values;
                }

                /*
                    Returns a tuple of originating index, property value, num values
                */
                std::tuple<size_t*, void*, size_t> get_property_values_device(uint64_t* desired_element_ids, size_t N_desired, std::string property_name, bool strict=true) {
                    if(!has_property_key(property_name)) {
                        std::stringstream sx;
                        sx << "Invalid property key " << property_name;
                        throw std::runtime_error(sx.str());
                    }

                    auto result = this->key_to_index_map[property_name].get_device(desired_element_ids, N_desired);
                    if(strict && std::get<2>(result) < N_desired) {
                        std::stringstream sx;
                        sx << "Some elements were missing a value for the property " << property_name;
                        throw std::runtime_error(sx.str());
                    }

                    return result;
                }

                inline bool has_property_key(std::string property_name) {
                    return this->key_to_index_map.find(property_name) != this->key_to_index_map.end();
                }

                inline gremlinxx::comparison::C get_dtype(std::string property_name) {
                    if(!has_property_key(property_name)) {
                        std::stringstream sx;
                        sx << "Invalid property key " << property_name;
                        throw std::runtime_error(sx.str());
                    }

                    return this->key_to_index_map[property_name].get_dtype();
                }
        };
    }
}