#include "structure/memory/TypeErasure.cuh"
#include "structure/memory/GPUHashTable.cuh"

#include "test/TestUtils.hpp"
#include <assert.h>
#include <cuda_runtime.h>
#include <tuple>

using namespace bitgraph::test;

void test_hash_basic();
void test_hash_double();

int main(int argc, char* argv[]) {
    try {
        test_hash_basic();
        test_hash_double();
    } catch(const std::exception& err) {
        std::cout << err.what() << "\n";
        return -1;
    }
}

void test_hash_basic() {
    std::vector<size_t> host_keys = {0, 1, 2, 3, 4, 5, 7};
    bitgraph::memory::TypeErasedVector keys(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::UINT64,
        host_keys.data(),
        host_keys.size(),
        false
    );

    std::vector<float> host_values = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.11f, 0.13f};
    bitgraph::memory::TypeErasedVector values(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::FLOAT32,
        host_values.data(),
        host_values.size(),
        false
    );

    bitgraph::memory::GPUHashTable table(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::UINT64,
        gremlinxx::comparison::C::FLOAT32
    );

    table.set(keys, values);

    bitgraph::memory::TypeErasedVector acquired_values;
    std::tie(acquired_values, std::ignore) = table.get(keys);
    std::vector<float> host_acquired_values(acquired_values.size());
    assert( keys.size() == acquired_values.size() );
    cudaMemcpy(host_acquired_values.data(), acquired_values.data(), sizeof(float) * keys.size(), cudaMemcpyDefault);

    assert_vector_equals(host_values, host_acquired_values);

}

void test_hash_double() {
    std::vector<size_t> host_keys = {0, 1, 2, 3, 4, 5, 7};
    bitgraph::memory::TypeErasedVector keys(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::UINT64,
        host_keys.data(),
        host_keys.size(),
        false
    );

    std::vector<float> host_values = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.11f, 0.13f};
    bitgraph::memory::TypeErasedVector values(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::FLOAT32,
        host_values.data(),
        host_values.size(),
        false
    );

    bitgraph::memory::GPUHashTable table(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::UINT64,
        gremlinxx::comparison::C::FLOAT32
    );

    table.set(keys, values);

    host_keys.clear();
    host_keys.push_back(102);
    host_keys.push_back(213);
    host_keys.push_back(77);

    host_values.clear();
    host_values.push_back(2.3f);
    host_values.push_back(0.14f);
    host_values.push_back(0.99f);

    keys = bitgraph::memory::TypeErasedVector(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::UINT64,
        host_keys.data(),
        host_keys.size(),
        false
    );

    values = bitgraph::memory::TypeErasedVector(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::FLOAT32,
        host_values.data(),
        host_values.size(),
        false
    );

    table.set(keys, values);

    host_keys.clear();
    host_keys.push_back(102);
    host_keys.push_back(1);
    host_keys.push_back(5);
    host_keys.push_back(7);
    host_keys.push_back(2);
    host_keys.push_back(0);
    host_keys.push_back(3);
    host_keys.push_back(77);
    host_keys.push_back(4);
    host_keys.push_back(213);

    keys = bitgraph::memory::TypeErasedVector(
        bitgraph::memory::memory_type::DEVICE,
        gremlinxx::comparison::C::UINT64,
        host_keys.data(),
        host_keys.size(),
        false
    );

    host_values.clear();
    host_values.push_back(2.3f);
    host_values.push_back(0.3f);
    host_values.push_back(0.11f);
    host_values.push_back(0.13f);
    host_values.push_back(0.5f);
    host_values.push_back(0.1f);
    host_values.push_back(0.7f);
    host_values.push_back(0.99f);
    host_values.push_back(0.9f);
    host_values.push_back(0.14f);

    bitgraph::memory::TypeErasedVector acquired_values;
    std::tie(acquired_values, std::ignore) = table.get(keys);
    std::vector<float> host_acquired_values(acquired_values.size());
    assert( keys.size() == 10 );
    assert( keys.size() == acquired_values.size() );
    cudaMemcpy(host_acquired_values.data(), acquired_values.data(), sizeof(float) * keys.size(), cudaMemcpyDefault);

    for(float x : host_acquired_values) std::cout << x << " ";
    std::cout << std::endl;

    assert_vector_equals(host_values, host_acquired_values);

}