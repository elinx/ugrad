#include "engine.hpp"

namespace ugrad {
Value::Value(double data) : _data(data), _grad(0.0f), _vis{false} {}
Value::Value(double data, vector<Value> children)
    : _data(data), _grad(0.0f), _children{children}, _vis{false} {}


void Value::build_topo(Value& val, vector<Value>& topo_order) {
  if (!val.visited()) {
    val.visited(true);
    for (auto child: val.children()) {
      build_topo(child, topo_order);
    }
    topo_order.insert(topo_order.begin(), val);
  }
}

}  // namespace ugrad
