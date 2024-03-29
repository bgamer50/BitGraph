#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include "gremlinxx/gremlinxx.h"
#include "bitgraph/structure/BitGraph.h"
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

NB_MODULE(PyBitGraph, m) {
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
            }
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
        })
        .def("add_vertices",
            [](
                bitgraph::BitGraph& bg,
                uint32_t size
            ) {
                bg.add_vertices(size);
        });
}