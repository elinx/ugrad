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
  Value relu() { return {std::max(0.0, _data), _children}; }

  void backward() {}

  friend ostream& operator<<(ostream& os, const Value& val) {
    os << "Value(data=" << val._data << ", grad=" << val._grad << ")";
    return os;
  }

 private:
  double _data;
  double _grad;
  vector<Value> _children;
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
