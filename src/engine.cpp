#include "engine.hpp"

namespace ugrad {
Value::Value(double data)
    : _data(data), _grad(0.0f), _vis{false}, _backward{[]() {}} {}
Value::Value(double data, vector<ValuePtr> children)
    : _data(data),
      _grad(0.0f),
      _children{children},
      _vis{false},
      _backward{[]() {}} {}

void Value::build_topo(ValuePtr val, vector<ValuePtr>& topo_order) {
  if (!val->visited()) {
    val->visited(true);
    for (auto child : val->children()) {
      build_topo(child, topo_order);
    }
    topo_order.insert(topo_order.begin(), val);
  }
}

void Value::clear_visit_mark(vector<ValuePtr>& topo_order) {
  for (auto& val: topo_order) {
    val->_vis = false;
  }
}

}  // namespace ugrad
