#include "engine.hpp"

namespace ugrad {
Value::Value(double data) : _data(data), _grad(0.0f) {}
}  // namespace ugrad
