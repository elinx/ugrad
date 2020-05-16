#ifndef __UGRAD_ENGINE_HPP__
#define __UGRAD_ENGINE_HPP__

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <ostream>
#include <vector>

namespace ugrad {

using std::make_shared;
using std::ostream;
using std::shared_ptr;
using std::vector;

struct Value;
using ValuePtr = shared_ptr<Value>;

struct Value : public std::enable_shared_from_this<Value> {
 public:
  Value(double data)
      : _data(data), _vis{false}, _grad(0.0f), _backward{[]() {}} {}

  Value(double data, vector<ValuePtr> children)
      : _data(data),
        _vis{false},
        _grad(0.0f),
        _children{children},
        _backward{[]() {}} {}

  double data() const { return _data; }
  void set_data(double data) { _data = data; }
  double grad() const { return _grad; }
  void set_grad(double grad) { _grad = grad; }
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
      self->_grad +=
          rhs->_data * std::pow(self->_data, rhs->_data - 1) * out->_grad;
    };
    return out;
  }

  ValuePtr pow(double exp) {
    auto rhs = make_shared<Value>(exp);
    return pow(rhs);
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

  void build_topo(ValuePtr val, vector<ValuePtr>& topo_order) {
    if (!val->visited()) {
      val->visited(true);
      for (auto child : val->children()) {
        build_topo(child, topo_order);
      }
      topo_order.insert(topo_order.begin(), val);
    }
  }

  void build_topo(vector<ValuePtr>& topo_order) {
    build_topo(shared_from_this(), topo_order);
  }

  void clear_visit_mark(vector<ValuePtr>& topo_order) {
    for (auto& val : topo_order) {
      val->_vis = false;
    }
  }

  friend ostream& operator<<(ostream& os, const Value& val) {
    os << "Value(data=" << val._data << ", grad=" << val._grad << ")";
    return os;
  }

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

inline ValuePtr operator+(ValuePtr lhs, double val) {
  auto rhs = make_shared<Value>(val);
  return lhs + rhs;
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

inline ValuePtr operator*(ValuePtr lhs, double val) {
  auto rhs = make_shared<Value>(val);
  return lhs * rhs;
}

inline ValuePtr operator*(double val, ValuePtr rhs) {
  auto lhs = make_shared<Value>(val);
  return lhs * rhs;
}

inline ValuePtr operator-(ValuePtr rhs) {
  return rhs * make_shared<Value>(-1.0);
}

inline ValuePtr operator-(ValuePtr lhs, ValuePtr rhs) { return lhs + (-rhs); }

inline ValuePtr operator/(ValuePtr lhs, ValuePtr rhs) {
  return lhs * rhs->pow(make_shared<Value>(-1));
}

inline ValuePtr operator/(ValuePtr lhs, double val) {
  auto rhs = make_shared<Value>(val);
  return lhs / rhs;
}

inline ValuePtr operator/(double val, ValuePtr rhs) {
  auto lhs = make_shared<Value>(val);
  return lhs / rhs;
}

}  // namespace ugrad
#endif  // __UGRAD_ENGINE_HPP__
