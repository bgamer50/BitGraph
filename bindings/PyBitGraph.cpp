#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include "gremlinxx/gremlinxx.h"
#include "bitgraph/structure/BitGraph.h"
#include "bitgraph/strategy/BitGraphSelectionStrategy.h"
#include "bitgraph/strategy/BitGraphStrategy.h"
#include "maelstrom/containers/vector.h"
#include "maelstrom/util/any_utils.h"

namespace nb = nanobind;
using namespace nb::literals;

maelstrom::dtype_t maelstrom_dtype_from_dlpack_dtype(nb::dlpack::dtype t) {
    if(t.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int)) {
        // signed integer
        switch(t.bits) {
            case 8: return maelstrom::int8;
            case 32: return maelstrom::int32;
            case 64: return maelstrom::int64;
        }
    } else if(t.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt)) {
        // unsigned integer
        switch(t.bits) {
            case 8: return maelstrom::uint8;
            case 32: return maelstrom::uint32;
            case 64: return maelstrom::uint64;
        }
    } else if(t.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) {
        // floating point
        switch(t.bits) {
            case 32: return maelstrom::float32;
            case 64: return maelstrom::float64;
        }
    }

    throw std::runtime_error("Unsupported data type");
}

maelstrom::storage maelstrom_storage_from_device_type(int32_t device_type) {
    switch(device_type) {
        case nb::device::cpu::value:
        case nb::device::none::value:
            return maelstrom::HOST;
        case nb::device::cuda::value:
            return maelstrom::DEVICE;
        case nb::device::cuda_managed::value:
            return maelstrom::MANAGED;
        case nb::device::cuda_host::value:
            return maelstrom::PINNED;
        default:
            throw std::runtime_error("Unsupported device");
    }
}

template<typename T, typename B, maelstrom::storage S, typename D>
nb::object t_maelstrom_to_py_ndarray(maelstrom::vector& vec) {
    if(vec.empty()) {
        const size_t shape[] = {0L};
        return nb::cast(
            nb::ndarray<B, const T, D>(
                nullptr,
                1,
                shape,
                nb::handle()
            )
        );
    }

    T* data = static_cast<T*>(vec.data());
    vec.disown();
    nb::capsule owner(data, [](void* ptr) noexcept {
        maelstrom::vector tmp(S, maelstrom::uint64, ptr, 1, true);
        tmp.own(); // will free memory once tmp goes out of scope
    });

    nb::ndarray<B, T, D> arr(
        data,
        {vec.size()},
        owner
    );

    auto cupy = nb::module_::import_("cupy");
    return arr.cast();
}

template<typename T>
nb::object maelstrom_to_py_ndarray_dispatch_backend(maelstrom::vector& vec) {
    switch(vec.get_mem_type()) {
        case maelstrom::HOST:
            return t_maelstrom_to_py_ndarray<T, nb::numpy, maelstrom::HOST, nb::device::cpu>(vec);
        case maelstrom::DEVICE:
            return t_maelstrom_to_py_ndarray<T, nb::cupy, maelstrom::DEVICE, nb::device::cuda>(vec);
        case maelstrom::MANAGED:
            return t_maelstrom_to_py_ndarray<T, nb::cupy, maelstrom::MANAGED, nb::device::cuda>(vec);
        case maelstrom::PINNED:
            return t_maelstrom_to_py_ndarray<T, nb::cupy, maelstrom::PINNED, nb::device::cuda>(vec);
    }

    throw std::runtime_error("invalid memory type for ndarray conversion");
}

nb::object maelstrom_to_py_ndarray(maelstrom::vector& vec) {
    if(vec.get_dtype().name == "string") {
        auto host_view_or_copy = maelstrom::as_host_vector(vec);
        nb::list out;

        auto dtype = vec.get_dtype();
        for(size_t k = 0; k < vec.size(); ++k) {
            char* loc = static_cast<char*>(vec.data()) + (maelstrom::size_of(dtype) * k);
            out.append(std::any_cast<std::string>(dtype.deserialize(loc)));
        }

        auto np = nb::module_::import_("numpy");
        return np.attr("asarray")(out);
    }

    switch(vec.get_dtype().prim_type) {
        case maelstrom::UINT64:
            return maelstrom_to_py_ndarray_dispatch_backend<uint64_t>(vec);
        case maelstrom::UINT32:
            return maelstrom_to_py_ndarray_dispatch_backend<uint32_t>(vec);
        case maelstrom::UINT8:
            return maelstrom_to_py_ndarray_dispatch_backend<uint8_t>(vec);
        case maelstrom::INT64:
            return maelstrom_to_py_ndarray_dispatch_backend<int64_t>(vec);
        case maelstrom::INT32:
            return maelstrom_to_py_ndarray_dispatch_backend<int32_t>(vec);
        case maelstrom::INT8:
            return maelstrom_to_py_ndarray_dispatch_backend<int8_t>(vec);
        case maelstrom::FLOAT64:
            return maelstrom_to_py_ndarray_dispatch_backend<double>(vec);
        case maelstrom::FLOAT32:
            return maelstrom_to_py_ndarray_dispatch_backend<float>(vec);
    }

    throw std::runtime_error("invalid primitive type");
}

NB_MODULE(pybitgraph, m) {
    nb::class_<bitgraph::BitGraph>(m, "BitGraph")
        .def("__init__",
            [](
                bitgraph::BitGraph* bg,
                std::string vertex_dtype,
                std::string edge_dtype,
                std::string structure_storage,
                std::string default_property_storage,
                std::string transfer_storage
            ) {
                auto vertex_m_dtype = maelstrom::dtype_string_mapping[vertex_dtype];
                auto edge_m_dtype = maelstrom::dtype_string_mapping[edge_dtype];

                auto structure_m_storage = maelstrom::storage_string_mapping[structure_storage];
                auto default_property_m_storage = maelstrom::storage_string_mapping[default_property_storage];
                auto transfer_m_storage = maelstrom::storage_string_mapping[transfer_storage];

                new (bg) bitgraph::BitGraph(
                    vertex_m_dtype,
                    edge_m_dtype,
                    structure_m_storage,
                    default_property_m_storage,
                    transfer_m_storage
                );
            },
            "vertex_dtype"_a,
            "edge_dtype"_a,
            "structure_storage"_a,
            "default_property_storage"_a,
            "transfer_storage"_a,
            "Constructs a BitGraph object representing a graph."
        )
        .def("add_edges", 
            [](
                bitgraph::BitGraph& bg,
                nb::ndarray<> src, 
                nb::ndarray<> dst,
                std::string label
            ) {
                auto src_dtype = maelstrom_dtype_from_dlpack_dtype(src.dtype());
                auto dst_dtype = maelstrom_dtype_from_dlpack_dtype(dst.dtype());

                if(src_dtype != dst_dtype) {
                    throw std::runtime_error("Source and destination data type must match!");
                }

                auto src_storage = maelstrom_storage_from_device_type(src.device_type());
                auto dst_storage = maelstrom_storage_from_device_type(src.device_type());

                maelstrom::vector src_view = maelstrom::vector(src_storage, src_dtype, src.data(), src.size(), true);
                maelstrom::vector dst_view = maelstrom::vector(dst_storage, dst_dtype, dst.data(), dst.size(), true);

                bg.add_edges(src_view, dst_view, label);
            },
            "src_array"_a,
            "dst_array"_a,
            "label"_a,
            "Adds edges to the graph."
        )
        .def("add_vertices",
            [](
                bitgraph::BitGraph& bg,
                size_t size
            ) {
                bg.add_vertices(size);
            },
            "num_vertices"_a,
            "Adds the given number of vertices to the graph."
        )
        .def("traversal",
            [](
                bitgraph::BitGraph& bg
            ) {
                nb::module_::import_("pygremlinxx");
                gremlinxx::GraphTraversalSource* source = new bitgraph::BitGraphTraversalSource(&bg);
                return source;
            }
        )
        .def("num_vertices",
             &bitgraph::BitGraph::num_vertices
        )
        .def("num_edges",
             &bitgraph::BitGraph::num_edges
        )
        .def("declare_vertex_property",
            [](
                bitgraph::BitGraph& bg,
                std::string name,
                std::string mem_type,
                std::string dtype,
                size_t initial_size
            ){
                auto m_storage = maelstrom::storage_string_mapping[mem_type];
                auto m_dtype = maelstrom::dtype_string_mapping[dtype];
                bg.declare_vertex_property(
                    name,
                    m_storage,
                    m_dtype,
                    initial_size
                );
            }
        )
        .def("make_vertex_embedding_index",
            &bitgraph::BitGraph::make_vertex_embedding_index
        )
        .def("set_vertex_properties",
            [](
                bitgraph::BitGraph& bg,
                std::string name,
                nb::ndarray<> vertices,
                nb::ndarray<> property_values
            ){
                auto m_vertices_dtype = maelstrom_dtype_from_dlpack_dtype(vertices.dtype());
                auto m_values_dtype = maelstrom_dtype_from_dlpack_dtype(property_values.dtype());

                auto m_vertices_storage = maelstrom_storage_from_device_type(vertices.device_type());
                auto m_values_storage = maelstrom_storage_from_device_type(property_values.device_type());

                if(m_vertices_dtype.prim_type != bg.get_vertex_dtype().prim_type) {
                    std::stringstream sx;
                    sx << "Vertex type of provided vertices (" << m_vertices_dtype.name << ") does not match the graph vertex type";
                    throw std::invalid_argument(sx.str());
                }

                maelstrom::vector m_vertices_view(
                    m_vertices_storage,
                    bg.get_vertex_dtype(),
                    vertices.data(),
                    vertices.size(),
                    true
                );

                maelstrom::vector m_values_view(
                    m_values_storage,
                    m_values_dtype,
                    property_values.data(),
                    property_values.size(),
                    true
                );

                bg.set_vertex_properties(name, m_vertices_view, m_values_view);
            }
        )
        .def("set_vertex_embeddings", 
            [](
                bitgraph::BitGraph& bg,
                std::string emb_name,
                nb::ndarray<> vertices,
                nb::ndarray<> embeddings
            ){
                auto m_vertices_storage = maelstrom_storage_from_device_type(vertices.device_type());
                maelstrom::vector m_vertices_view(
                    m_vertices_storage,
                    bg.get_vertex_dtype(),
                    vertices.data(),
                    vertices.size(),
                    true
                );

                auto m_embeddings_storage = maelstrom_storage_from_device_type(embeddings.device_type());
                auto m_embeddings_dtype = maelstrom_dtype_from_dlpack_dtype(embeddings.dtype());
                maelstrom::vector m_embeddings_view(
                    m_embeddings_storage,
                    m_embeddings_dtype,
                    embeddings.data(),
                    embeddings.size(),
                    true
                );

                bg.set_vertex_embeddings(
                    emb_name,
                    m_vertices_view,
                    m_embeddings_view
                );
            }
        )
        .def("set_vertex_embeddings",
            [](
                bitgraph::BitGraph& bg,
                std::string emb_name,
                size_t v_start,
                size_t v_end,
                nb::ndarray<> embeddings
            ){
                auto m_embeddings_storage = maelstrom_storage_from_device_type(embeddings.device_type());
                auto m_embeddings_dtype = maelstrom_dtype_from_dlpack_dtype(embeddings.dtype());
                maelstrom::vector m_embeddings_view(
                    m_embeddings_storage,
                    m_embeddings_dtype,
                    embeddings.data(),
                    embeddings.size(),
                    true
                );

                bg.set_vertex_embeddings(
                    emb_name,
                    v_start,
                    v_end,
                    m_embeddings_view
                );
            }
        )
        .def("subgraph_coo",
            [](
                bitgraph::BitGraph& bg,
                nb::ndarray<> edges
            ){
                auto m_edges_storage = maelstrom_storage_from_device_type(edges.device_type());
                auto m_edges_dtype = maelstrom_dtype_from_dlpack_dtype(edges.dtype());
                maelstrom::vector m_edges_view(
                    m_edges_storage,
                    m_edges_dtype,
                    edges.data(),
                    edges.size(),
                    true
                );

                maelstrom::vector src;
                maelstrom::vector dst;
                maelstrom::vector vertices;
                std::tie(vertices, src, dst) = bg.get_subgraph_coo(m_edges_view);
                
                nb::dict d;
                d["src"] = maelstrom_to_py_ndarray(src);
                d["dst"] = maelstrom_to_py_ndarray(dst);
                d["vid"] = maelstrom_to_py_ndarray(vertices);

                return d;
            }
        )
        .def("get_vertex_property_names", &bitgraph::BitGraph::get_vertex_property_names)
        .def("get_edge_property_names", &bitgraph::BitGraph::get_edge_property_names)
        .def_ro_static("BitGraphSelectionStrategy", &bitgraph::BitGraphSelectionStrategy)
        .def_ro_static("BitGraphStrategy", &bitgraph::BitGraphStrategy);
}