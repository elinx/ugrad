#ifndef __UGRAD_ENGINE_HPP__
#define __UGRAD_ENGINE_HPP__

#include <algorithm>
#include <ostream>
#include <vector>

namespace ugrad {

using std::ostream;
using std::vector;

class Value {
 public:
  Value(double data);
  Value(double data, vector<Value> children);

  double data() const { return _data; }
  double grad() const { return _grad; }
  const vector<Value>& children() const { return _children; }
  void children(const vector<Value>& children) { _children = children; }
  bool visited() { return _vis; }
  void visited(bool status) { _vis = status; }
  Value relu() { return {std::max(0.0, _data), _children}; }

  void backward() {
    _grad = 1.0;
    auto topo_order = build_topo();
    for (auto val: topo_order) {
      val.backward();
    }
  }

  vector<Value> build_topo() {
    vector<Value> topo_order;
    visited(false);
    build_topo(topo_order);
    return topo_order;
  }

  friend ostream& operator<<(ostream& os, const Value& val) {
    os << "Value(data=" << val._data << ", grad=" << val._grad << ")";
    return os;
  }

 private:
  void build_topo(Value& val, vector<Value>& topo_order);
  void build_topo(vector<Value>& topo_order) { build_topo(*this, topo_order); }

 private:
  double _data;
  double _grad;
  vector<Value> _children;
  bool _vis;
};

inline Value operator+(const Value& lhs, const Value& rhs) {
  auto out = Value(lhs.data() + rhs.data(), {lhs, rhs});
  return out;
}

inline Value operator-(const Value& lhs, const Value& rhs) {
  auto out = Value(lhs.data() - rhs.data(), {lhs, rhs});
  return out;
}

inline Value operator*(const Value& lhs, const Value& rhs) {
  auto out = Value(lhs.data() * rhs.data(), {lhs, rhs});
  return out;
}

inline Value operator/(const Value& lhs, const Value& rhs) {
  auto out = Value(lhs.data() / rhs.data(), {lhs, rhs});
  return out;
}

inline Value operator-(const Value& rhs) {
  return {-rhs.data(), rhs.children()};
}

inline Value& operator+=(Value& lhs, const Value& rhs) {
  lhs = lhs + rhs;
  return lhs;
}

inline Value& operator-=(Value& lhs, const Value& rhs) {
  lhs = lhs - rhs;
  return lhs;
}

inline Value& operator*=(Value& lhs, const Value& rhs) {
  lhs = lhs * rhs;
  return lhs;
}

inline Value& operator/=(Value& lhs, const Value& rhs) {
  lhs = lhs / rhs;
  return lhs;
}

}  // namespace ugrad
#endif  // __UGRAD_ENGINE_HPP__
