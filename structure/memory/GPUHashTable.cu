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

         /*
            This method sets the given keys to the given values.
        */
        void GPUHashTable::set(TypeErasedVector& new_keys, TypeErasedVector& new_values) {
            if(new_keys.size() != new_values.size()) throw std::runtime_error("Keys and values must be of same length!");
            size_t new_size = this->num_elements_in_table + new_keys.size();
            size_t expected_fill_factor = new_size / this->keys.size();
            if(this->keys.size() == 0 || expected_fill_factor > this->max_fill_factor) {
                this->resize(new_size * 2 + 1);
            }
            
            size_t* insert_indices = hash_insert(
                this->keys.data(),
                new_keys.data(),
                gremlinxx::comparison::C_size[new_keys.get_dtype()],
                this->keys.size(),
                new_keys.size()
            );

            bool should_rehash = thrust::any_of(
                thrust::device,
                thrust::device_pointer_cast<size_t>(insert_indices),
                thrust::device_pointer_cast<size_t>(insert_indices) + new_keys.size(),
                bitgraph::memory::is_max_val<size_t>()
            );
            if(should_rehash) {
                std::cout << "rehashing before limit!" << std::endl;
                cudaFree(insert_indices);
                cudaCheckErrors("free insert_indices");
                this->resize(this->keys.size() * 1.4 + 1);

                return this->set(new_keys, new_values);
            }

            TypeErasedVector new_keys_copy(new_keys);
            TypeErasedVector new_values_copy(new_values);

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

            // Elements where adjacent difference is 0 will not be inserted,
            // and will be instead inserted in the next recursive call.
            // Element 0 is always valid so we set it so to avoid the 0 edge case
            thrust::adjacent_difference(
                thrust::device_pointer_cast<size_t>(insert_indices) + 1,
                thrust::device_pointer_cast<size_t>(insert_indices) + new_values_copy.size(),
                thrust::device_pointer_cast<size_t>(diffs) + 1
            );
            cudaMemset(diffs, 0xff, 1);

            size_t remaining_data_size = set_scatter(
                this->keys.data(),
                new_keys_copy.data(),
                new_keys_copy.get_dtype(),
                this->values.data(),
                new_values_copy.data(),
                new_values_copy.get_dtype(),
                insert_indices,
                diffs,
                new_values_copy.size()
            );

            cudaFree(insert_indices);
            cudaFree(diffs);
            cudaCheckErrors("Free insert indices, diffs");

            this->num_elements_in_table += new_keys.size() - remaining_data_size;

            if(remaining_data_size > 0) {
                new_keys_copy.resize(remaining_data_size);
                new_values_copy.resize(remaining_data_size);
                this->set(
                    new_keys_copy,
                    new_values_copy
                );
            }
        }

        TypeErasedVector GPUHashTable::get(TypeErasedVector& desired_keys, bool strict) {
            if(desired_keys.get_dtype() != this->keys.get_dtype()) {
                throw std::runtime_error("Provided key type does not match key type in hash table!");
            }

            TypeErasedVector retrieved_values = get_scan(
                this->keys,
                this->values,
                desired_keys
            );

            if(strict && (retrieved_values.size() != desired_keys.size())) {
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