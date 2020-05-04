#ifndef __UGRAD_ENGINE_HPP__
#define __UGRAD_ENGINE_HPP__

#include <algorithm>
#include <functional>
#include <memory>
#include <ostream>
#include <vector>
#include <cmath>

namespace ugrad {

using std::make_shared;
using std::ostream;
using std::shared_ptr;
using std::vector;

struct Value;
using ValuePtr = shared_ptr<Value>;

struct Value : public std::enable_shared_from_this<Value> {
 public:
  Value(double data);
  Value(double data, vector<ValuePtr> children);

  double data() const { return _data; }
  double grad() const { return _grad; }
  const vector<ValuePtr>& children() const { return _children; }
  void children(const vector<ValuePtr>& children) { _children = children; }
  bool visited() { return _vis; }
  void visited(bool status) { _vis = status; }

  ValuePtr relu() {
    auto out = make_shared<Value>(std::max(0.0, _data),
                                  vector<ValuePtr>{shared_from_this()});
    out->_backward = [out, self = shared_from_this()]() {
      self->_grad += (out->_data > 0) * out->_grad;
    };
    return out;
  }

  ValuePtr pow(ValuePtr rhs) {
    auto out = make_shared<Value>(std::pow(_data, rhs->_data),
                                  vector<ValuePtr>{shared_from_this()});
    out->_backward = [out, self = shared_from_this(), rhs]() {
      self->_grad += rhs->_data * std::pow(self->_data, rhs->_data - 1) * out->_grad;
    };
    return out;
  }

  void backward() {
    _grad = 1.0;
    auto topo_order = build_topo();
    for (auto val : topo_order) {
      val->_backward();
    }
  }

  vector<ValuePtr> build_topo() {
    vector<ValuePtr> topo_order;
    build_topo(topo_order);
    clear_visit_mark(topo_order);
    return topo_order;
  }

  friend ostream& operator<<(ostream& os, const Value& val) {
    os << "Value(data=" << val._data << ", grad=" << val._grad << ")";
    return os;
  }

  void build_topo(ValuePtr val, vector<ValuePtr>& topo_order);
  void build_topo(vector<ValuePtr>& topo_order) {
    build_topo(shared_from_this(), topo_order);
  }
  void clear_visit_mark(vector<ValuePtr>& topo_order);

  double _data;
  bool _vis;
  double _grad;
  vector<ValuePtr> _children;
  std::function<void()> _backward;
};

inline ValuePtr operator+(ValuePtr lhs, ValuePtr rhs) {
  auto out =
      make_shared<Value>(lhs->data() + rhs->data(), vector<ValuePtr>{lhs, rhs});
  out->_backward = [out, lhs, rhs]() {
    lhs->_grad += out->_grad;
    rhs->_grad += out->_grad;
  };
  return out;
}

inline ValuePtr operator*(ValuePtr lhs, ValuePtr rhs) {
  auto out =
      make_shared<Value>(lhs->data() * rhs->data(), vector<ValuePtr>{lhs, rhs});
  out->_backward = [out, lhs, rhs]() {
    lhs->_grad += rhs->data() * out->_grad;
    rhs->_grad += lhs->data() * out->_grad;
  };
  return out;
}

inline ValuePtr operator-(ValuePtr rhs) {
  return rhs * make_shared<Value>(-1.0);
}

inline ValuePtr operator-(ValuePtr lhs, ValuePtr rhs) { return lhs + (-rhs); }

inline ValuePtr operator/(ValuePtr lhs, ValuePtr rhs) {
  // auto out =
  //     make_shared<Value>(lhs->data() / rhs->data(), vector<ValuePtr>{lhs, rhs});
  // out->_backward = [out, lhs, rhs]() {
  //   lhs->_grad += (1.0 / rhs->data()) * out->_grad;
  //   rhs->_grad += -1.0 * lhs->data() / (rhs->data() * rhs->data()) * out->_grad;
  // };
  // return out;
  return lhs * rhs->pow(make_shared<Value>(-1));
}

}  // namespace ugrad
#endif  // __UGRAD_ENGINE_HPP__
