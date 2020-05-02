#include "engine.hpp"

namespace ugrad {
Value::Value(double data) : _data(data), _grad(0.0f) {}
Value::Value(double data, vector<Value> children)
    : _data(data), _grad(0.0f), _children{children} {}
}  // namespace ugrad
