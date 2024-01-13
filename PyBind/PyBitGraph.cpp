#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include "gremlinxx/gremlinxx.h"
#include "bitgraph/structure/BitGraph.h"
#include "maelstrom/containers/vector.h"
#include "maelstrom/util/any_utils.h"

namespace nb = nanobind;
using namespace nb::literals;

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
        );
        //.def("add_edges", &bitgraph::add_edges)
    // virtual maelstrom::vector add_edges(maelstrom::vector& from_vertices, maelstrom::vector& to_vertices, std::string label);
}