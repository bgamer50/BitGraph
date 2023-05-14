#pragma once

#include "gremlinxx/gremlinxx.h"
#include "structure/memory/StringUtils.h"

#include <iostream>
#include <optional>

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
                std::string name;

                // Creates a blank vector with the given memory type and data type.
                TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype);

                // Default constructor; creates a blank device vector of FLOAT64 dtype
                TypeErasedVector();

                // Creates a vector of size N unitialized values of the given data type and given memory type.
                TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype, size_t N);

                // Creates a vector corresponding to the provided data.  If view=true then this vector is only a view
                // over the provided data.  If view=false then this vector will own a copy of the provided data.
                TypeErasedVector(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C dtype, void* data, size_t N, bool view=true);

                TypeErasedVector(TypeErasedVector& orig, bool view);

                TypeErasedVector(TypeErasedVector& orig);

                ~TypeErasedVector();

                TypeErasedVector(TypeErasedVector&& other) noexcept;
                
                TypeErasedVector& operator=(TypeErasedVector&& other) noexcept;

                inline bool is_view() { return this->view; }

                void push_back();

                void reserve(size_t N);

                void insert(); // single insert, range insert

                void insert(size_t ix_start, TypeErasedVector& new_elements);

                inline void erase(){} // single erase, range erase
                inline void get(){} // single get, range get

                /*
                    Empties the contents of this vector and frees any reserved memory.
                */
                void clear();

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

                /*
                    The currently-owning vector will no longer own its data, and it will
                    not be cleaned up when this vector is deleted.  Essentially, this
                    vector becomes a view.
                */
                inline void disown() {
                    if(this->view) throw std::runtime_error("Vector does not own data!");
                    else this->view = true;
                }

                void resize(size_t N);

                /*
                    Creates a copy of this vector with the given memory type.
                */
                TypeErasedVector to(bitgraph::memory::memory_type mem_type);

        };

        /*
            Creates a new vector with the same memory type and data type as the given vector.

            Arguments
            ---------
            other: TypeErasedVector&
                Reference to an existing type erased vector.  Will determine the memory type
                and data type of the new vector based on this existing vector.
            N: size_t
                Initial size of the new vector.  Defaults to 0.
            
            Returns
            -------
            TypeErasedVector
                The new type erased vector with the same memory type and data type as the
                vector that was passed in, and the provided initial size (or an empty vector
                if the size argument was not provided).
        */
        TypeErasedVector make_vector_like(TypeErasedVector& other, size_t N=0);

        /*
            Creates a type erased vector from a std::vector of anys.  Obviously, the new
            vector cannot share memory with the original vector of anys, even if it is
            on host memory.

            Arguments
            ---------
            anys: std::vector<boost::any>&
                Reference to a std::vector of anys that will be copied into a new type erased vector.
            mem_type: bitgraph::memory::memory_type
                The memory type of the new type erased vector.
            
            Returns
            -------
            TypeErasedVector
                The new type erased vector created from the vector of anys.
        */
        TypeErasedVector make_vector_from_anys(std::vector<boost::any>& anys, bitgraph::memory::memory_type mem_type=bitgraph::memory::memory_type::HOST, StringIndex* string_index=nullptr);

        /*
            Creates a std::vector of anys from a TypeErasedVector.  Obviously, the new
            vector cannot share memory with the original type erased vector, even if it
            was on host memory.

            Arguments
            ---------
            vec: TypeErasedVector&
                Reference to a type erased vector that will be converted to a vector of anys.
            
            Returns
            -------
            std::vector<boost::any>
                A new std::vector of anys.
        */
        std::vector<boost::any> vector_to_anys(TypeErasedVector& vec, StringIndex* string_index=nullptr);

        /*
            Creates a new type erased vector that is a view of the provided std::vector.
            Since the new vector is a view, it will share memory with the original std::vector,
            with ownership belonging to the original std::vector.

            Arguments
            ---------
            typed_vector: std::vector<T>&
                Reference to the typed vector to return a type erased view of.
            
            Returns
            -------
            TypeErasedVector
                A non-owning view of the original std::vector.  The original std::vector
                will continue to own the memory.
        */
        template<typename T>
        TypeErasedVector make_viewing_vector_from_typed(std::vector<T>& typed_vector);

        bool are_mem_types_compatible(bitgraph::memory::memory_type t1, bitgraph::memory::memory_type t2);
    }
}
