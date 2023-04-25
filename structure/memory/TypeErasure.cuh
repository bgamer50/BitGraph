#pragma once

#include "gremlinxx/gremlinxx.h"

#include <iostream>

namespace bitgraph {
    namespace memory {
        enum memory_type {HOST=0, MANAGED=1, DEVICE=2, PINNED=3};

        class TypeErasedVector {
            private:
                void* data_ptr;
                size_t filled_size;
                size_t reserved_size;
                gremlinxx::comparison::C dtype;
                bitgraph::memory::memory_type mem_type;
                bool view;

                void* alloc(size_t N);

                void dealloc(void* ptr);

                void copy(void* src, void* dst, size_t size);
            
            public:
                // Creates a blank vector with the given memory type and data type.
                TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype);

                // Default constructor; creates a blank device vector of FLOAT64 dtype
                TypeErasedVector();

                // Creates a vector of size N unitialized values of the given data type and given memory type.
                TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype, size_t N);

                // Creates a vector corresponding to the provided data.  If view=true then this vector is only a view
                // over the provided data.  If view=false then this vector will own a copy of the provided data.
                TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype, void* data, size_t N, bool view=true);

                TypeErasedVector(TypeErasedVector& orig);

                ~TypeErasedVector();

                TypeErasedVector(TypeErasedVector&& other) noexcept;
                
                TypeErasedVector& operator=(TypeErasedVector&& other) noexcept;

                inline bool is_view() { return this->view; }

                void push_back();

                void reserve(size_t N);

                void insert(); // single insert, range insert

                void insert(size_t ix_start, TypeErasedVector& new_elements);

                void erase(){} // single erase, range erase
                void get(){} // single get, range get

                /*
                    Copies the vector to the host and prints it.
                */
                void print();
                
                inline size_t size() { return this->filled_size; }

                inline void* data() {
                    return this->data_ptr;
                }

                inline gremlinxx::comparison::C get_dtype() { return this->dtype; }

                inline bitgraph::memory::memory_type get_mem_type() { return this->mem_type; }

                // This currently-viewing vector will take ownership of the data and it will
                // follow the lifecycle of this vector.
                inline void own() {
                    if(this->view) this->view = false;
                    else throw std::runtime_error("Vector already owns data!"); 
                }

                inline void disown() {
                    if(this->view) throw std::runtime_error("Vector does not own data!");
                    else this->view = true;
                }

                void resize(size_t N);

        };

        TypeErasedVector make_vector_like(TypeErasedVector& other, size_t N=0);

    }
}
