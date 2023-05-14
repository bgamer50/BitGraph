#include "structure/memory/TypeErasure.cuh"

#include <sstream>
#include <cuda_runtime.h>
#include "util/cuda_utils.cuh"

namespace bitgraph {
    namespace memory {
        
        void* TypeErasedVector::alloc(size_t N) {
            size_t dtype_size = gremlinxx::comparison::C_size[this->dtype];

            switch(this->mem_type) {
                case bitgraph::memory::memory_type::HOST: {
                    return static_cast<void*>(new char[N * dtype_size]);
                }
                case bitgraph::memory::memory_type::DEVICE: {
                    void* ptr;
                    cudaMalloc(&ptr, dtype_size * N);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("TypeErasedVector alloc device memory");
                    return ptr;
                }
                case bitgraph::memory::memory_type::MANAGED: {
                    void* ptr;
                    cudaMallocManaged(&ptr, dtype_size * N);
                    cudaDeviceSynchronize();
                    std::stringstream sx;
                    sx << "TypeErasedVector alloc managed memory (" << this->name << ")";
                    cudaCheckErrors(sx.str());
                    return ptr;
                }
                case bitgraph::memory::memory_type::PINNED: {
                    void* ptr;
                    cudaMallocHost(&ptr, dtype_size * N);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("TypeErasedVector alloc pinned memory");
                    return ptr;
                }
            }

            throw std::runtime_error("Invalid memory type provided to TypeErasedVector alloc()");
        }

        void TypeErasedVector::dealloc(void* ptr) {
            switch(this->mem_type) {
                case bitgraph::memory::memory_type::HOST: {
                    delete static_cast<char*>(ptr);
                    return;
                }
                case bitgraph::memory::memory_type::MANAGED: {
                    cudaFree(ptr);
                    cudaDeviceSynchronize();
                    std::stringstream sx;
                    sx << "TypeErasedVector dealloc managed memory (" << this->name << ")";
                    cudaCheckErrors(sx.str());
                    return;
                }
                case bitgraph::memory::memory_type::DEVICE: {
                    cudaFree(ptr);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("TypeErasedVector dealloc device memory");
                    return;
                }
                case bitgraph::memory::memory_type::PINNED: {
                    cudaFreeHost(ptr);
                    cudaDeviceSynchronize();
                    cudaCheckErrors("TypeErasedVector dealloc pinned memory");
                    return;
                }
            }

            throw std::runtime_error("Invalid memory type provided to TypeErasedVector dealloc");
        }

        // Copies from src (first arg) to dst (second arg) using cudaMemcpy.
        void TypeErasedVector::copy(void* src, void* dst, size_t size) {
            cudaMemcpy(dst, src, gremlinxx::comparison::C_size[this->dtype] * size, cudaMemcpyDefault);
            cudaCheckErrors("TypeErasedVector copy");
        }

        void TypeErasedVector::clear() {
            if(this->view) throw std::runtime_error("Cannot clear a view!");

            this->dealloc(this->data_ptr);
            this->filled_size = 0;
            this->reserved_size = 0;
            this->data_ptr = nullptr;
        }

        // Creates a blank vector with the given memory type and data type.
        TypeErasedVector::TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype) {
            this->mem_type = mem_type;
            this->dtype = dtype;
            this->filled_size = 0;
            this->reserved_size = 0;
            this->data_ptr = nullptr;
            this->view = false;
        }

        // Default constructor; creates a blank device vector of FLOAT64 dtype
        TypeErasedVector::TypeErasedVector()
        : TypeErasedVector(bitgraph::memory::memory_type::DEVICE, gremlinxx::comparison::C::FLOAT64) {}

        // Creates a vector of size N unitialized values of the given data type and given memory type.
        TypeErasedVector::TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype, size_t N) {
            this->mem_type = mem_type;
            this->dtype = dtype;
            this->filled_size = 0;
            this->reserved_size = 0;
            this->data_ptr = nullptr;
            this->view = false;

            this->resize(N);
        }

        // Creates a vector corresponding to the provided data.  If view=true then this vector is only a view
        // over the provided data.  If view=false then this vector will own a copy of the provided data.
        TypeErasedVector::TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype, void* data, size_t N, bool view) {
            this->mem_type = mem_type;
            this->dtype = dtype;
            this->view = view;
            this->reserved_size = 0;

            if(this->view) { 
                this->data_ptr = data; 
                this->filled_size = N;
                this->reserved_size = N;
            }
            else {
                this->resize(N);
                this->copy(data, this->data_ptr, N);
            }
        }

        TypeErasedVector::TypeErasedVector(TypeErasedVector& orig, bool view)
        : TypeErasedVector(
            orig.mem_type,
            orig.dtype,
            orig.data_ptr,
            orig.filled_size,
            view
        ) {}

        TypeErasedVector::TypeErasedVector(TypeErasedVector& orig) {
            this->mem_type = orig.mem_type;
            this->dtype = orig.dtype;
            this->filled_size = orig.filled_size;
            this->reserved_size = 0;
            this->view = false;

            this->resize(orig.filled_size);
            this->copy(orig.data_ptr, this->data_ptr, orig.filled_size);                    
        }

        TypeErasedVector::~TypeErasedVector() {
            if(this->data_ptr != nullptr && !this->view) {
                this->dealloc(this->data_ptr);
            }
        }

        TypeErasedVector::TypeErasedVector(TypeErasedVector&& other) noexcept {
            this->data_ptr = std::move(other.data_ptr);
            this->filled_size = std::move(other.filled_size);
            this->reserved_size = std::move(other.reserved_size);
            this->dtype = std::move(other.dtype);
            this->mem_type = std::move(other.mem_type);
            this->view = other.view;
            other.view = true;
        }
        
        TypeErasedVector& TypeErasedVector::operator=(TypeErasedVector&& other) noexcept {
            this->data_ptr = std::move(other.data_ptr);
            this->filled_size = std::move(other.filled_size);
            this->reserved_size = std::move(other.reserved_size);
            this->dtype = std::move(other.dtype);
            this->mem_type = std::move(other.mem_type);
            this->view = other.view;
            other.view = true;

            return *this;
        }

        void TypeErasedVector::push_back() {
            throw std::runtime_error("push_back unimplemented");
        }

        void TypeErasedVector::reserve(size_t N) {
            if(N < this->filled_size) throw std::runtime_error("Cannot reserve fewer elements than size!");
            
            if(N <= this->reserved_size) return;

            size_t old_filled_size = this->filled_size;
            this->resize(N);
            this->filled_size = old_filled_size;
        }

        void TypeErasedVector::insert() {
            throw std::runtime_error("insert unimplemented");
        }

        void TypeErasedVector::insert(size_t ix_start, TypeErasedVector& new_elements) {
            if(this->view) throw std::runtime_error("Cannot insert into a view!");
            if(this->dtype != new_elements.dtype) throw std::runtime_error("Data type of inserting vector must match!");

            size_t old_size = this->size();
            size_t new_size = old_size + new_elements.size();
            
            void* new_data = this->data_ptr;
            if(new_size > reserved_size) {
                new_data = this->alloc(new_size);
                this->reserved_size = new_size;   
            }

            size_t elements_to_copy = old_size - ix_start;
            size_t element_size = gremlinxx::comparison::C_size[this->dtype];

            if(elements_to_copy > 0) {
                this->copy(
                    static_cast<char*>(this->data_ptr) + (element_size * ix_start),
                    static_cast<char*>(new_data) + (element_size * (ix_start + new_elements.size())),
                    elements_to_copy
                );
            }

            this->copy(
                new_elements.data(),
                static_cast<char*>(new_data) + (element_size * ix_start),
                new_elements.size()
            );

            if(this->data_ptr != new_data) {
                if(ix_start > 0) {
                    this->copy(
                        this->data_ptr,
                        new_data,
                        ix_start
                    );
                }

                cudaCheckErrors("check errors before dealloc");
                this->dealloc(this->data_ptr);
                this->data_ptr = new_data;
            }

            this->filled_size = new_size;
        }

        /*
            Copies the vector to the host and prints it.
        */
        void TypeErasedVector::print() {
            std::vector<size_t> h_data(this->size());
            cudaMemcpy(h_data.data(), this->data(), gremlinxx::comparison::C_size[this->dtype] * this->size(), cudaMemcpyDefault);
            cudaDeviceSynchronize();
            cudaCheckErrors("copy to host");
            for(auto x : h_data) std::cout << x << " ";
            std::cout << std::endl;
        }

        void TypeErasedVector::resize(size_t N) {
            if(this->view) throw std::runtime_error("Cannot resize a view!");

            bool empty = (this->reserved_size == 0);
            
            // Don't resize if there is already enough space reserved
            if(N <= reserved_size) {
                this->filled_size = N;
                return;
            }

            void* new_data = this->alloc(N);
            if(!empty) {
                this->copy(this->data_ptr, new_data, this->filled_size);
                this->dealloc(this->data_ptr);
            }
            
            this->data_ptr = new_data;
            this->filled_size = N;
            this->reserved_size = N;
        }

        TypeErasedVector TypeErasedVector::to(bitgraph::memory::memory_type mem_type) {
            auto new_vec = TypeErasedVector(
                mem_type,
                this->get_dtype()
            );

            new_vec.insert(0, *this);
            return new_vec;
        }

        TypeErasedVector make_vector_like(TypeErasedVector& other, size_t N) {
            return TypeErasedVector(
                other.get_mem_type(),
                other.get_dtype(),
                N
            );
        }

        TypeErasedVector make_vector_from_anys(std::vector<boost::any>& anys, bitgraph::memory::memory_type mem_type, StringIndex* string_index) {
            gremlinxx::comparison::C dtype = gremlinxx::comparison::from_any(anys.front());
            size_t value_size = gremlinxx::comparison::C_size[dtype];
            unsigned char* data = new unsigned char[value_size * anys.size()];

            bool is_string = (dtype == gremlinxx::comparison::C::STRING);
            size_t i = 0;
            if(is_string) {
                if(string_index == nullptr) throw std::runtime_error("Cannot store type-erased strings without string index!");
                auto converted_values = string_index->from_cpu_anys(anys);
                for(boost::any& b : converted_values) {
                    void* ptr = static_cast<void*>(data + (i * value_size));
                    gremlinxx::comparison::move_any(b, ptr);
                    ++i;
                }
                    
            } else {
                for(boost::any& b : anys) {
                    void* ptr = static_cast<void*>(data + (i * value_size));
                    gremlinxx::comparison::move_any(b, ptr);
                    ++i;
                }
            }

            auto host_values = TypeErasedVector(
                bitgraph::memory::memory_type::HOST,
                dtype,
                static_cast<void*>(data),
                anys.size(),
                false
            );

            if(mem_type == bitgraph::memory::memory_type::HOST) {
                return host_values;
            }
            
            return host_values.to(mem_type);
        }

        std::vector<boost::any> vector_to_anys(TypeErasedVector& vec, StringIndex* string_index) {
            size_t N = vec.size();
            auto dtype = vec.get_dtype();
            bool is_string = (dtype == gremlinxx::comparison::C::STRING);
            size_t value_size = gremlinxx::comparison::C_size[dtype];

            std::vector<boost::any> output_vector;
            output_vector.resize(N);

            if(vec.get_mem_type() == bitgraph::memory::memory_type::HOST) {
                unsigned char* data = static_cast<unsigned char*>(vec.data());
                for(size_t k = 0; k < N; ++k) { 
                    output_vector[k] = gremlinxx::comparison::move_void(static_cast<void*>(data + (k * value_size)), dtype);
                }
            } else {
                auto vec_host = vec.to(bitgraph::memory::memory_type::HOST);
                unsigned char* data = static_cast<unsigned char*>(vec_host.data());
                for(size_t k = 0; k < N; ++k) { 
                    output_vector[k] = gremlinxx::comparison::move_void(static_cast<void*>(data + (k * value_size)), dtype);
                }
            }

            if(is_string) {
                if(string_index == nullptr) throw std::runtime_error("String index required to convert type erased strings");
                return string_index->from_gpu_anys(output_vector);
            } 

            return output_vector;
        }

        template<typename T>
        TypeErasedVector make_viewing_vector_from_typed(std::vector<T>& typed_vector) {
            return TypeErasedVector(
                bitgraph::memory::memory_type::HOST,
                gremlinxx::comparison::C_TYPEID<T>(),
                static_cast<void*>(typed_vector.data()),
                typed_vector.size(),
                true
            );
        }

        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<uint64_t>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<uint32_t>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<uint8_t>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<int64_t>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<int32_t>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<int8_t>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<float>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<double>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<std::string>& typed_vector);
        template
        TypeErasedVector make_viewing_vector_from_typed(std::vector<Vertex*>& typed_vector);

        bool are_mem_types_compatible(bitgraph::memory::memory_type t1, bitgraph::memory::memory_type t2) {
            if(t1 == t2) return true;
            else if(t1 == bitgraph::memory::memory_type::PINNED || t2 == bitgraph::memory::memory_type::PINNED) return false;
            else if(t1 == bitgraph::memory::memory_type::MANAGED || t2 == bitgraph::memory::memory_type::MANAGED) return true;
            return false;
        }
    }
}