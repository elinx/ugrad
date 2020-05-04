#ifndef __UGRAD_ENGINE_HPP__
#define __UGRAD_ENGINE_HPP__

#include <algorithm>
#include <functional>
#include <ostream>
#include <vector>

namespace ugrad {

using std::ostream;
using std::reference_wrapper;
using std::vector;

struct Value {
 public:
  using ValueRef = reference_wrapper<Value>;

  Value(double data);
  Value(double data, vector<ValueRef> children);

  double data() const { return _data; }
  double grad() const { return _grad; }
  const vector<ValueRef>& children() const { return _children; }
  void children(const vector<ValueRef>& children) { _children = children; }
  bool visited() { return _vis; }
  void visited(bool status) { _vis = status; }
  Value relu() { return {std::max(0.0, _data), _children}; }

  Value& operator+=(const Value& rhs) {
    auto out = Value(this->data() + rhs.data(), {*this, rhs});
    out._backward = [&out] () {
      out._children[0].get()._grad += out._grad;
      out._children[1].get()._grad += out._grad;
    };
    *this = out;
    return *this;
  }

  Value& operator-=(const Value& rhs) {
    // lhs = lhs - rhs;
    return *this;
  }

  Value& operator*=(const Value& rhs) {
    // lhs = lhs * rhs;
    return *this;
  }

  Value& operator/=(const Value& rhs) {
    lhs = lhs / rhs;
    return *this;
  }

  void backward() {
    _grad = 1.0;
    auto topo_order = build_topo();
    for (auto val: topo_order) {
      val.get()._backward();
    }
  }

  vector<ValueRef> build_topo() {
    vector<ValueRef> topo_order;
    build_topo(topo_order);
    clear_visit_mark(topo_order);
    return topo_order;
  }

  friend ostream& operator<<(ostream& os, const Value& val) {
    os << "Value(data=" << val._data << ", grad=" << val._grad << ")";
    return os;
  }

  void build_topo(Value& val, vector<ValueRef>& topo_order);
  void build_topo(vector<ValueRef>& topo_order) { build_topo(*this, topo_order); }
  void clear_visit_mark(vector<ValueRef>& topo_order);

  double _data;
  bool _vis;
  double _grad;
  vector<ValueRef> _children;
  std::function<void()> _backward;
};

inline Value operator+(Value& lhs, Value& rhs) {
  auto out = Value(lhs.data() + rhs.data(), {lhs, rhs});
  out._backward = [&out] () {
    out._children[0].get()._grad += out._grad;
    out._children[1].get()._grad += out._grad;
  };
  return out;
}

inline Value operator-(Value& lhs, Value& rhs) {
  auto out = Value(lhs.data() - rhs.data(), {lhs, rhs});
  return out;
}

inline Value operator*(Value& lhs, Value& rhs) {
  auto out = Value(lhs.data() * rhs.data(), {lhs, rhs});
  out._backward = [&out] () {
    out._children[0].get()._grad += out._children[1].get().data() * out._grad;
    out._children[1].get()._grad += out._children[0].get().data() * out._grad;
  };
  return out;
}

inline Value operator/(Value& lhs, Value& rhs) {
  auto out = Value(lhs.data() / rhs.data(), {lhs, rhs});
  return out;
}

inline Value operator-(const Value& rhs) {
  return {-rhs.data(), rhs.children()};
}


}  // namespace ugrad
#endif  // __UGRAD_ENGINE_HPP__
