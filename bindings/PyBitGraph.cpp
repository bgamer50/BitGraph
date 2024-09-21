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

                // Assume these are on the host
                maelstrom::vector src_view = maelstrom::vector(maelstrom::HOST, src_dtype, src.data(), src.size(), true);
                maelstrom::vector dst_view = maelstrom::vector(maelstrom::HOST, dst_dtype, dst.data(), dst.size(), true);

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
        .def("set_vertex_properties",
            [](
                bitgraph::BitGraph& bg,
                std::string name,
                nb::ndarray<> vertices,
                nb::ndarray<> property_values
            ){
                auto m_vertices_dtype = maelstrom_dtype_from_dlpack_dtype(vertices.dtype());
                auto m_values_dtype = maelstrom_dtype_from_dlpack_dtype(property_values.dtype());

                if(m_vertices_dtype.prim_type != bg.get_vertex_dtype().prim_type) {
                    std::stringstream sx;
                    sx << "Vertex type of provided vertices (" << m_vertices_dtype.name << ") does not match the graph vertex type";
                    throw std::invalid_argument(sx.str());
                }

                maelstrom::vector m_vertices_view(
                    maelstrom::HOST,
                    bg.get_vertex_dtype(),
                    vertices.data(),
                    vertices.size(),
                    true
                );
                //m_vertices_view.pin();

                maelstrom::vector m_values_view(
                    maelstrom::HOST,
                    m_values_dtype,
                    property_values.data(),
                    property_values.size(),
                    true
                );
                //m_values_view.pin();

                bg.set_vertex_properties(name, m_vertices_view, m_values_view);
                //m_vertices_view.unpin();
                //m_values_view.unpin();
            }
        )
        .def("set_vertex_embeddings", 
            [](
                bitgraph::BitGraph& bg,
                std::string emb_name,
                nb::ndarray<> vertices,
                nb::ndarray<> embeddings
            ){
                maelstrom::vector m_vertices_view(
                    maelstrom::HOST,
                    bg.get_vertex_dtype(),
                    vertices.data(),
                    vertices.size(),
                    true
                );

                auto m_embeddings_dtype = maelstrom_dtype_from_dlpack_dtype(embeddings.dtype());
                maelstrom::vector m_embeddings_view(
                    maelstrom::HOST,
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
        .def("get_vertex_property_names", &bitgraph::BitGraph::get_vertex_property_names)
        .def("get_edge_property_names", &bitgraph::BitGraph::get_edge_property_names)
        .def_ro_static("BitGraphSelectionStrategy", &bitgraph::BitGraphSelectionStrategy)
        .def_ro_static("BitGraphStrategy", &bitgraph::BitGraphStrategy);
}