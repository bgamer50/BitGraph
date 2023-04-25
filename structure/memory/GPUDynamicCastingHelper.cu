#include "structure/memory/GPUDynamicCastingHelper.cuh"
#include "util/cuda_utils.cuh"
#include "structure/memory/ThrustUtils.cuh"

namespace bitgraph {
    namespace memory {
        
        /*
            Copies casted data from the input array into the
            output array.
        */
        template<typename T, typename U>
        void array_cast(T* input_array, U* output_array, size_t array_length) {
            thrust::device_ptr<T> input_array_tptr = thrust::device_pointer_cast(input_array);
            thrust::device_ptr<U> output_array_tptr = thrust::device_pointer_cast(output_array);

            caster<T, U> caster_inst;
            thrust::copy(
                thrust::make_transform_iterator(input_array, caster_inst),
                thrust::make_transform_iterator(input_array + array_length, caster_inst),
                output_array_tptr
            );
        }        

        template<typename T>
        void
        array_cast_inner_wrapper(T* input_array, void* output_array, size_t array_length, gremlinxx::comparison::C output_dtype) {
            switch(output_dtype) {
                case gremlinxx::comparison::C::INT8:
                    return array_cast<T, int8_t>(input_array, static_cast<int8_t*>(output_array), array_length);
                case gremlinxx::comparison::C::INT32:
                    return array_cast<T, int32_t>(input_array, static_cast<int32_t*>(output_array), array_length);
                case gremlinxx::comparison::C::INT64:
                    return array_cast<T, int64_t>(input_array, static_cast<int64_t*>(output_array), array_length);
                case gremlinxx::comparison::C::UINT8:
                    return array_cast<T, uint8_t>(input_array, static_cast<uint8_t*>(output_array), array_length);
                case gremlinxx::comparison::C::UINT32:
                    return array_cast<T, uint32_t>(input_array, static_cast<uint32_t*>(output_array), array_length);
                case gremlinxx::comparison::C::UINT64:
                    return array_cast<T, uint64_t>(input_array, static_cast<uint64_t*>(output_array), array_length);
                case gremlinxx::comparison::C::FLOAT32:
                    return array_cast<T, float>(input_array, static_cast<float*>(output_array), array_length);
                case gremlinxx::comparison::C::FLOAT64:
                    return array_cast<T, double>(input_array, static_cast<double*>(output_array), array_length);
            }

            throw std::runtime_error("illegal dtype");
        }

        void
        array_cast_outer_wrapper(void* input_array, void* output_array, size_t array_length, gremlinxx::comparison::C input_dtype, gremlinxx::comparison::C output_dtype) {
            switch(output_dtype) {
                case gremlinxx::comparison::C::INT8:
                    return array_cast_inner_wrapper(static_cast<int8_t*>(input_array), output_array, array_length, output_dtype);
                case gremlinxx::comparison::C::INT32:
                    return array_cast_inner_wrapper(static_cast<int32_t*>(input_array), output_array, array_length, output_dtype);
                case gremlinxx::comparison::C::INT64:
                    return array_cast_inner_wrapper(static_cast<int64_t*>(input_array), output_array, array_length, output_dtype);
                case gremlinxx::comparison::C::UINT8:
                    return array_cast_inner_wrapper(static_cast<uint8_t*>(input_array), output_array, array_length, output_dtype);
                case gremlinxx::comparison::C::UINT32:
                    return array_cast_inner_wrapper(static_cast<uint32_t*>(input_array), output_array, array_length, output_dtype);
                case gremlinxx::comparison::C::UINT64:
                    return array_cast_inner_wrapper(static_cast<uint64_t*>(input_array), output_array, array_length, output_dtype);
                case gremlinxx::comparison::C::FLOAT32:
                    return array_cast_inner_wrapper(static_cast<float*>(input_array), output_array, array_length, output_dtype);
                case gremlinxx::comparison::C::FLOAT64:
                    return array_cast_inner_wrapper(static_cast<double*>(input_array), output_array, array_length, output_dtype);
            }

            throw std::runtime_error("illegal dtype");
        }

        gremlinxx::comparison::C get_output_dtype(gremlinxx::comparison::C left_dtype, gremlinxx::comparison::C right_dtype) {
            if(left_dtype == right_dtype) return right_dtype;

            gremlinxx::comparison::C dtype;
            if(left_dtype == gremlinxx::comparison::C::UINT8) {
                if(right_dtype == gremlinxx::comparison::C::UINT32) dtype = gremlinxx::comparison::C::UINT32;
                else if(right_dtype == gremlinxx::comparison::C::UINT64) dtype = gremlinxx::comparison::C::UINT64;
                else if(right_dtype == gremlinxx::comparison::C::INT8) dtype = gremlinxx::comparison::C::INT32;
                else if(right_dtype == gremlinxx::comparison::C::INT32) dtype = gremlinxx::comparison::C::INT32;
                else if(right_dtype == gremlinxx::comparison::C::INT64) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT32) dtype = gremlinxx::comparison::C::FLOAT32;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT32) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else if(left_dtype == gremlinxx::comparison::C::UINT32) {
                if(right_dtype == gremlinxx::comparison::C::UINT8) dtype = gremlinxx::comparison::C::UINT32;
                else if(right_dtype == gremlinxx::comparison::C::UINT64) dtype = gremlinxx::comparison::C::UINT64;
                else if(right_dtype == gremlinxx::comparison::C::INT8) dtype = gremlinxx::comparison::C::INT32;
                else if(right_dtype == gremlinxx::comparison::C::INT32) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::INT64) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT32) dtype = gremlinxx::comparison::C::FLOAT32;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT64) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else if(left_dtype == gremlinxx::comparison::C::UINT64) {
                if(right_dtype == gremlinxx::comparison::C::UINT8 || right_dtype == gremlinxx::comparison::C::UINT32) dtype = gremlinxx::comparison::C::UINT64;
                else if(right_dtype == gremlinxx::comparison::C::INT8 || right_dtype == gremlinxx::comparison::C::INT32 
                        || right_dtype == gremlinxx::comparison::C::INT64) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT32 || right_dtype == gremlinxx::comparison::C::FLOAT64) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else if(left_dtype == gremlinxx::comparison::C::INT8) {
                if(right_dtype == gremlinxx::comparison::C::UINT8) dtype = gremlinxx::comparison::C::INT32;
                else if(right_dtype == gremlinxx::comparison::C::UINT32) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::INT32) dtype = gremlinxx::comparison::C::INT32;
                else if(right_dtype == gremlinxx::comparison::C::INT64) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT32) dtype = gremlinxx::comparison::C::FLOAT32;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT64) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else if(dtype == gremlinxx::comparison::C::INT32) {
                if(right_dtype == gremlinxx::comparison::C::INT8 || right_dtype == gremlinxx::comparison::C::UINT8) dtype = gremlinxx::comparison::C::INT32;
                else if(right_dtype == gremlinxx::comparison::C::UINT32 || right_dtype == gremlinxx::comparison::C::INT64) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT32) dtype = gremlinxx::comparison::C::FLOAT32;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT64) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else if(dtype == gremlinxx::comparison::C::INT64) {
                if(right_dtype == gremlinxx::comparison::C::UINT8 || right_dtype == gremlinxx::comparison::C::UINT32 
                    || right_dtype == gremlinxx::comparison::C::INT8 || right_dtype == gremlinxx::comparison::C::INT32) dtype = gremlinxx::comparison::C::INT64;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT32 || right_dtype == gremlinxx::comparison::C::FLOAT64) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else if(dtype == gremlinxx::comparison::C::FLOAT32) {
                if(right_dtype == gremlinxx::comparison::C::UINT8 || right_dtype == gremlinxx::comparison::C::INT8 
                    || right_dtype == gremlinxx::comparison::C::UINT32 || right_dtype == gremlinxx::comparison::C::INT32) dtype = gremlinxx::comparison::C::FLOAT32;
                else if(right_dtype == gremlinxx::comparison::C::FLOAT64) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else if(dtype == gremlinxx::comparison::C::FLOAT64) {
                if(right_dtype == gremlinxx::comparison::C::UINT8 || right_dtype == gremlinxx::comparison::C::UINT32 
                    || right_dtype == gremlinxx::comparison::C::UINT64 || right_dtype == gremlinxx::comparison::C::INT8
                    || right_dtype == gremlinxx::comparison::C::INT32 || right_dtype == gremlinxx::comparison::C::INT64
                    || right_dtype == gremlinxx::comparison::C::FLOAT32) dtype = gremlinxx::comparison::C::FLOAT64;
                else {
                    std::stringstream sx;
                    sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                    throw std::runtime_error(sx.str());
                }
            }
            else {
                std::stringstream sx;
                sx << "Cannot mix types " << gremlinxx::comparison::C_to_string[dtype] << " and " << gremlinxx::comparison::C_to_string[right_dtype];
                throw std::runtime_error(sx.str());
            }

            return dtype;
        }

        /*
            Appends the elements in right to the end of left, upcasting dtype if necessary.
            Frees left and right.
        */
        std::tuple<void*, gremlinxx::comparison::C, size_t> device_array_combine(void* left, gremlinxx::comparison::C left_dtype, size_t left_length, void* right, gremlinxx::comparison::C right_dtype, size_t right_length) {
            auto dtype = get_output_dtype(left_dtype, right_dtype);

            size_t left_memsize = gremlinxx::comparison::C_size[dtype] * left_length;
            size_t right_memsize = gremlinxx::comparison::C_size[dtype] * right_length;

            char* new_data;
            cudaMalloc(&new_data, left_memsize + right_memsize);
            cudaDeviceSynchronize();
            cudaCheckErrors("allocate new data");

            array_cast_outer_wrapper(
                left,
                new_data,
                left_length,
                left_dtype,
                dtype
            );
            cudaFree(left);
            cudaDeviceSynchronize();
            cudaCheckErrors("copy left to new data");
            
            array_cast_outer_wrapper(
                right,
                new_data + left_memsize,
                right_length,
                right_dtype,
                dtype
            );
            cudaFree(right);
            cudaDeviceSynchronize();
            cudaCheckErrors("copy right to new data");

            return std::make_tuple(static_cast<void*>(new_data), left_dtype, left_length + right_length);
        }

        // Template Instantiations
        // FIXME these may not be necessary

        template
        void array_cast(int8_t* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(int8_t* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(int8_t* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(int8_t* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(int8_t* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(int8_t* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(int8_t* input_array, float* output_array, size_t array_length);
        template
        void array_cast(int8_t* input_array, double* output_array, size_t array_length);

        template
        void array_cast(int32_t* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(int32_t* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(int32_t* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(int32_t* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(int32_t* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(int32_t* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(int32_t* input_array, float* output_array, size_t array_length);
        template
        void array_cast(int32_t* input_array, double* output_array, size_t array_length);

        template
        void array_cast(int64_t* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(int64_t* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(int64_t* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(int64_t* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(int64_t* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(int64_t* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(int64_t* input_array, float* output_array, size_t array_length);
        template
        void array_cast(int64_t* input_array, double* output_array, size_t array_length);

        template
        void array_cast(uint8_t* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(uint8_t* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(uint8_t* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(uint8_t* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(uint8_t* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(uint8_t* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(uint8_t* input_array, float* output_array, size_t array_length);
        template
        void array_cast(uint8_t* input_array, double* output_array, size_t array_length);

        template
        void array_cast(uint32_t* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(uint32_t* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(uint32_t* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(uint32_t* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(uint32_t* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(uint32_t* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(uint32_t* input_array, float* output_array, size_t array_length);
        template
        void array_cast(uint32_t* input_array, double* output_array, size_t array_length);

        template
        void array_cast(uint64_t* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(uint64_t* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(uint64_t* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(uint64_t* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(uint64_t* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(uint64_t* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(uint64_t* input_array, float* output_array, size_t array_length);
        template
        void array_cast(uint64_t* input_array, double* output_array, size_t array_length);

        template
        void array_cast(float* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(float* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(float* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(float* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(float* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(float* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(float* input_array, float* output_array, size_t array_length);
        template
        void array_cast(float* input_array, double* output_array, size_t array_length);

        
        template
        void array_cast(double* input_array, int8_t* output_array, size_t array_length);
        template
        void array_cast(double* input_array, int32_t* output_array, size_t array_length);
        template
        void array_cast(double* input_array, int64_t* output_array, size_t array_length);
        template
        void array_cast(double* input_array, uint8_t* output_array, size_t array_length);
        template
        void array_cast(double* input_array, uint32_t* output_array, size_t array_length);
        template
        void array_cast(double* input_array, uint64_t* output_array, size_t array_length);
        template
        void array_cast(double* input_array, float* output_array, size_t array_length);
        template
        void array_cast(double* input_array, double* output_array, size_t array_length);

    }
}