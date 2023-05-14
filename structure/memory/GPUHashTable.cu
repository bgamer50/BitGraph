#include <tuple>

#include "structure/memory/GPUHashTable.cuh"

#include "structure/memory/primitives/hash_primitives/HashFilterValid.cuh"
#include "structure/memory/primitives/hash_primitives/HashGetScan.cuh"
#include "structure/memory/primitives/hash_primitives/HashInsert.cuh"
#include "structure/memory/primitives/hash_primitives/HashSetScatter.cuh"
#include "structure/memory/primitives/hash_primitives/HashSetSort.cuh"

#include "structure/memory/ThrustUtils.cuh"

#include <cuda_runtime.h>
#include <util/cuda_utils.cuh>

namespace bitgraph {
    namespace memory {

        GPUHashTable::GPUHashTable(bitgraph::memory::memory_type mem_type, gremlinxx::comparison::C key_dtype, gremlinxx::comparison::C val_dtype, size_t reserved_size) {
            this->keys = TypeErasedVector(mem_type, key_dtype);
            this->values = TypeErasedVector(mem_type, val_dtype);
            this->num_elements_in_table = 0;
            if(reserved_size > 0) {
                this->keys.reserve(reserved_size);
                this->values.reserve(reserved_size);
                cudaMemset(
                    this->keys.data(),
                    static_cast<char>(0xff),
                    reserved_size * gremlinxx::comparison::C_size[key_dtype]
                );
            }
        }

        GPUHashTable::GPUHashTable()
        : GPUHashTable(bitgraph::memory::memory_type::MANAGED, gremlinxx::comparison::C::UINT64, gremlinxx::comparison::C::UINT64, 0) {}

         /*
            This method sets the given keys to the given values.
        */
        void GPUHashTable::set(TypeErasedVector& new_keys, TypeErasedVector& new_values) {
            if(new_keys.get_dtype() != this->keys.get_dtype()) throw std::runtime_error("New keys datatype must match existing keys datatype");
            if(new_values.get_dtype() != this->values.get_dtype()) throw std::runtime_error("New values datatype must match existing values datatype");
            if(new_keys.size() != new_values.size()) throw std::runtime_error("Keys and values must be of same length!");

            // Create views of the new keys and values so we can mess with them while still allowing pass by reference
            TypeErasedVector new_keys_view = TypeErasedVector(new_keys, true);
            TypeErasedVector new_values_view = TypeErasedVector(new_values, true);

            // If we need to move the keys or values to the correct memory location, technically they're no longer views (which is ok)
            // They will always be automatically cleaned up appropriately
            if(new_keys_view.get_mem_type() != this->keys.get_mem_type()) new_keys_view = new_keys_view.to(this->keys.get_mem_type());
            if(new_values_view.get_mem_type() != this->values.get_mem_type()) new_values_view = new_values_view.to(this->values.get_mem_type());

            size_t new_key;
            cudaMemcpy(&new_key, new_keys.data(), sizeof(size_t), cudaMemcpyDefault);
            if(new_key == 13) std::cout << "setting key #13" << std::endl;
            
            size_t new_size = this->num_elements_in_table + new_keys_view.size();
            size_t expected_fill_factor = new_size / this->keys.size();
            
            if(this->keys.size() == 0 || expected_fill_factor > this->max_fill_factor) {
                this->resize(new_size * 2 + 1);
            }
            
            size_t* insert_indices = hash_insert(
                this->keys.data(),
                new_keys_view.data(),
                gremlinxx::comparison::C_size[new_keys_view.get_dtype()],
                this->keys.size(),
                new_keys_view.size()
            );

            if(new_key == 13) {
                TypeErasedVector ii(
                    bitgraph::memory::memory_type::DEVICE,
                    gremlinxx::comparison::C::UINT64,
                    insert_indices,
                    new_keys_view.size(),
                    true
                );

                std::cout << "ii for 13: ";
                ii.print();
            }

            bool should_rehash = thrust::any_of(
                thrust::device,
                thrust::device_pointer_cast<size_t>(insert_indices),
                thrust::device_pointer_cast<size_t>(insert_indices) + new_keys_view.size(),
                bitgraph::memory::is_max_val<size_t>()
            );

            if(should_rehash) {
                std::cerr << "warning: rehashing before limit!" << std::endl;
                cudaFree(insert_indices);
                cudaCheckErrors("free insert_indices");
                this->resize(this->keys.size() * 1.4 + 1);

                return this->set(new_keys_view, new_values_view);
            }

            TypeErasedVector new_keys_copy(new_keys_view);
            TypeErasedVector new_values_copy(new_values_view);

            set_sort(
                insert_indices,
                new_keys_copy.data(),
                new_keys_copy.get_dtype(),
                new_values_copy.data(),
                new_values_copy.get_dtype(),
                new_values_copy.size()
            );

            size_t* diffs;
            cudaMalloc(&diffs, sizeof(size_t) * new_values.size());
            cudaDeviceSynchronize();
            cudaCheckErrors("Allocate diffs");

            // Elements where adjacent difference is 0 will not be inserted,
            // and will be instead inserted in the next recursive call.
            // Element 0 is always valid so we set it so to avoid the 0 edge case
            thrust::adjacent_difference(
                thrust::device_pointer_cast<size_t>(insert_indices) + 1,
                thrust::device_pointer_cast<size_t>(insert_indices) + new_values_copy.size(),
                thrust::device_pointer_cast<size_t>(diffs) + 1
            );
            cudaMemset(diffs, 0xff, 1);
            cudaDeviceSynchronize();
            cudaCheckErrors("memset diffs");

            size_t new_num_elements;
            size_t remaining_data_size;
            std::tie(new_num_elements, remaining_data_size) = set_scatter(
                this->keys.data(),
                new_keys_copy.data(),
                new_keys_copy.get_dtype(),
                this->values.data(),
                new_values_copy.data(),
                new_values_copy.get_dtype(),
                insert_indices,
                diffs,
                new_values_copy.size(),
                this->keys.size()
            );
            this->num_elements_in_table = new_num_elements;

            
            size_t val;
            cudaMemcpy(&val, static_cast<size_t*>(this->keys.data())+108, sizeof(size_t), cudaMemcpyDefault);
            if(new_key == 13) std::cout << "value of hashed #13: " << val << std::endl;
            

            cudaFree(insert_indices);
            cudaFree(diffs);
            cudaCheckErrors("Free insert indices, diffs");

            if(remaining_data_size > 0) {
                new_keys_copy.resize(remaining_data_size);
                new_values_copy.resize(remaining_data_size);
                this->set(
                    new_keys_copy,
                    new_values_copy
                );
            }
        }

        std::pair<TypeErasedVector, TypeErasedVector> GPUHashTable::get(TypeErasedVector& desired_keys, bool strict, bool return_indices) {
            if(strict && return_indices) throw std::runtime_error("Returning indices is not compatible with strict mode");

            if(desired_keys.get_dtype() != this->keys.get_dtype()) {
                throw std::runtime_error("Provided key type does not match key type in hash table!");
            }

            // Create views of the new keys and values so we can mess with them while still allowing pass by reference
            TypeErasedVector desired_keys_view(desired_keys, true);

            if(this->keys.size() == 0) {
                if(strict) throw std::runtime_error("Table is empty!");
                else return std::make_pair(
                    bitgraph::memory::make_vector_like(this->values),
                    TypeErasedVector()
                );
            }

            // If we need to move the desired keys to the correct memory location, technically it is no longer a view (which is ok)
            // They will always be automatically cleaned up appropriately
            if(desired_keys_view.get_mem_type() != this->keys.get_mem_type()) desired_keys_view = desired_keys_view.to(this->keys.get_mem_type());

            auto retrieved_values = get_scan(
                this->keys,
                this->values,
                desired_keys_view,
                return_indices
            );

            if(strict && (retrieved_values.first.size() != desired_keys_view.size())) {
                
                
                std::cout << "desired keys: ";
                desired_keys.print();

                throw std::runtime_error("Some values were not found in the hash table for the provided keys");
            }

            return retrieved_values;
        }

        void GPUHashTable::resize(size_t new_table_size) {
            // remove all the empty entries from keys, values
            filter_valid_values(this->keys, this->values);

            TypeErasedVector old_keys = std::move(this->keys);
            TypeErasedVector old_values = std::move(this->values);

            if(old_keys.size() != this->num_elements_in_table || old_values.size() != this->num_elements_in_table) {
                std::stringstream ss;
                ss << "A bookkeeping error was encountered while attempting to resize a hash table." << std::endl;
                ss << "Expected to find " << this->num_elements_in_table << " valid elements" << std::endl;
                ss << "but got " << old_keys.size() << " keys and " << old_values.size() << " values.";
                throw std::runtime_error(ss.str());
            }

            this->keys = bitgraph::memory::make_vector_like(old_keys, new_table_size);
            this->values = bitgraph::memory::make_vector_like(old_values, new_table_size);
            this->num_elements_in_table = 0;

            cudaMemset(
                this->keys.data(),
                static_cast<char>(0xff),
                new_table_size * gremlinxx::comparison::C_size[old_keys.get_dtype()]
            );
            
            if(old_keys.size() > 0) {
                this->set(old_keys, old_values);
            }
        }

    }
}