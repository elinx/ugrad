#ifndef __UGRAD_ENGINE_HPP__
#define __UGRAD_ENGINE_HPP__
#include <ostream>
namespace ugrad {

using std::ostream;

class Value {
 public:
  Value(double data);

  double data() { return _data; }
  double grad() { return _grad; }
  Value relu() { return *this; }

  Value operator+=(const Value& rhs) {
    this->_data += rhs._data;
    return *this;
  }

  Value operator-=(const Value& rhs) {
    this->_data -= rhs._data;
    return *this;
  }

  Value operator*=(const Value& rhs) {
    this->_data *= rhs._data;
    return *this;
  }

  Value operator/=(const Value& rhs) {
    this->_data /= rhs._data;
    return *this;
  }

  friend ostream& operator<<(ostream& os, const Value& val) {
    os << "Value(data=" << val._data << ", grad=" << val._grad
      << ")";
    return os;
  }

 private:
  double _data;
  double _grad;
};

inline Value operator+(Value lhs, const Value& rhs) {
  lhs += rhs;
  return lhs;
}

inline Value operator-(Value lhs, const Value& rhs) {
  lhs -= rhs;
  return lhs;
}

inline Value operator*(Value lhs, const Value& rhs) {
  lhs *= rhs;
  return lhs;
}

inline Value operator/(Value lhs, const Value& rhs) {
  lhs /= rhs;
  return lhs;
}

inline Value operator-(const Value& rhs) {
  return rhs;
}

}  // namespace ugrad

#endif  // __UGRAD_ENGINE_HPP__
