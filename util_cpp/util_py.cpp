
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "sum_tree_base.h"
#include "prio_experience.h"

namespace py = pybind11;


PYBIND11_MODULE(SharkUtil, m)
{

     py::class_<SumTreeBase>(m, "SumTreeBase")
        .def(py::init<int>())

        .def("add", &SumTreeBase::add)
        .def("update_value", &SumTreeBase::update_value)

        .def("get_value", &SumTreeBase::get_value)
        .def("tree_value", &SumTreeBase::tree_value)

        .def("find", &SumTreeBase::find,  py::arg("value"), py::arg("norm")=true )
        
        .def("size", &SumTreeBase::get_size)
        .def_property_readonly("capacity", &SumTreeBase::capacity)
        .def_property_readonly("tree_level", &SumTreeBase::tree_level)
        ;


    py::class_<PrioExperienceBase>(m, "PrioritizedExperienceBase")
        .def(py::init<int, int>())

        .def("add_c", &PrioExperienceBase::add)
        .def("sample_c", &PrioExperienceBase::sample)
        .def("update_priority_c", &PrioExperienceBase::update_priority)

        .def("size", &PrioExperienceBase::get_size)
        .def("__len__", &PrioExperienceBase::get_size)

        .def_property_readonly("capacity", &PrioExperienceBase::capacity)
        .def_property_readonly("batch_size", &PrioExperienceBase::batch_size)
        ;
}

















