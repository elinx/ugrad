#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <sstream>
#include <ugrad/engine.hpp>
#include <ugrad/nn.hpp>

namespace py = pybind11;
using ugrad::Value;
using ugrad::ValuePtr;

PYBIND11_MODULE(pyugrad, m) {
  py::class_<Value, std::shared_ptr<Value>>(m, "Value")
      .def(py::init<double>())
      .def("data", &Value::data)
      .def("grad", &Value::grad)
      .def("backward", &Value::backward)
      .def("__neg__", [](ValuePtr lhs) { return -lhs; })
      .def("__add__", [](ValuePtr lhs, ValuePtr rhs) { return lhs + rhs; })
      .def("__radd__", [](ValuePtr lhs, ValuePtr rhs) { return lhs + rhs; })
      .def("__sub__", [](ValuePtr lhs, ValuePtr rhs) { return lhs - rhs; })
      .def("__rsub__", [](ValuePtr lhs, ValuePtr rhs) { return rhs - lhs; })
      .def("__mul__", [](ValuePtr lhs, ValuePtr rhs) { return lhs * rhs; })
      .def("__rmul__", [](ValuePtr lhs, ValuePtr rhs) { return lhs * rhs; })
      .def("__truediv__", [](ValuePtr lhs, ValuePtr rhs) { return lhs / rhs; })
      .def("__rtruediv__", [](ValuePtr lhs, ValuePtr rhs) { return rhs / lhs; })
      .def("__repr__", [](const Value& val) {
        std::stringstream ss;
        ss << val;
        return ss.str();
      });
}
