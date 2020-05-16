#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <sstream>

#include <ugrad/engine.hpp>
#include <ugrad/nn.hpp>

namespace py = pybind11;
using ugrad::Value;
using ugrad::ValuePtr;
using ugrad::Module;
using ugrad::Neuron;
using ugrad::Layer;
using ugrad::MLP;

PYBIND11_MODULE(pyugrad, m) {
  py::class_<Value, std::shared_ptr<Value>>(m, "Value")
      .def(py::init<double>())
      .def(py::init<int>())
      .def_property("data", &Value::data, &Value::set_data)
      .def_property("grad", &Value::grad, &Value::set_grad)
      .def("backward", &Value::backward)
      .def("relu", &Value::relu)
      .def("__neg__", [](ValuePtr lhs) { return -lhs; })
      .def("__add__", [](ValuePtr lhs, ValuePtr rhs) { return lhs + rhs; })
      .def("__add__", [](ValuePtr lhs, double rhs) { return lhs + rhs; })
      .def("__radd__", [](ValuePtr lhs, ValuePtr rhs) { return lhs + rhs; })
      .def("__radd__", [](ValuePtr lhs, double rhs) { return lhs + rhs; })
      .def("__sub__", [](ValuePtr lhs, ValuePtr rhs) { return lhs - rhs; })
      .def("__rsub__", [](ValuePtr lhs, ValuePtr rhs) { return rhs - lhs; })
      .def("__mul__", [](ValuePtr lhs, ValuePtr rhs) { return lhs * rhs; })
      .def("__mul__", [](ValuePtr lhs, double rhs) { return lhs * rhs; })
      .def("__rmul__", [](ValuePtr lhs, ValuePtr rhs) { return lhs * rhs; })
      .def("__rmul__", [](ValuePtr lhs, double rhs) { return lhs * rhs; })
      .def("__truediv__", [](ValuePtr lhs, ValuePtr rhs) { return lhs / rhs; })
      .def("__truediv__", [](ValuePtr lhs, double rhs) { return lhs / rhs; })
      .def("__rtruediv__", [](ValuePtr lhs, ValuePtr rhs) { return rhs / lhs; })
      .def("__rtruediv__", [](ValuePtr lhs, double rhs) { return rhs / lhs; })
      .def("__pow__", [](ValuePtr lhs, ValuePtr rhs) { return lhs->pow(rhs); })
      .def("__pow__", [](ValuePtr lhs, double rhs) { return lhs->pow(rhs); })
      .def("__repr__", [](const Value& val) {
        std::stringstream ss;
        ss << val;
        return ss.str();
      });

  py::class_<Module>(m, "Module")
    .def(py::init<>())
    .def("zero_grad", &Module::zero_grad)
    .def("parameters", &Module::parameters);

  py::class_<Neuron, Module>(m, "Neuron")
    .def(py::init<size_t>())
    .def("__call__", &Neuron::operator())
    .def_property_readonly("parameters", &Neuron::parameters)
    .def("__repr__", [](const Neuron& neuron) {
        std::stringstream ss;
        ss << neuron;
        return ss.str();
    });

  py::class_<Layer, Module>(m, "Layer")
    .def(py::init<size_t, size_t>())
    .def("__call__", &Layer::operator())
    .def_property_readonly("parameters", &Layer::parameters)
    .def("__repr__", [](const Layer& layer) {
        std::stringstream ss;
        ss << layer;
        return ss.str();
    });

  py::class_<MLP, Module>(m, "MLP")
    .def(py::init<size_t, std::vector<size_t>>())
    .def("__call__", &MLP::operator())
    .def("parameters", &MLP::parameters)
    .def("__repr__", [](const MLP& mlp) {
        std::stringstream ss;
        ss << mlp;
        return ss.str();
    });
}
