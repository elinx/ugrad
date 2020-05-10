#ifndef __UGRAD_NN_HPP__
#define __UGRAD_NN_HPP__

#include <initializer_list>
#include <random>
#include <vector>
#include <string>
#include <sstream>

#include <ugrad/engine.hpp>

namespace ugrad {

struct Module {
  virtual ~Module() {}
  void zero_grad() {
    for (auto p: parameters()) {
      p->_grad = 0;
    }
  }
  virtual vector<ValuePtr> parameters() { return {}; }
};

struct Neuron : public Module {
  Neuron(size_t in_nr, bool non_linear = true, bool is_test = false)
      : _w{}, _b{make_shared<Value>(0.0)}, _non_linear{non_linear} {
    fill_weights(in_nr, !is_test);
  }
  ~Neuron() {}

  struct UniformRandomGenerator {
    UniformRandomGenerator() : _rng(std::random_device{}()), _dist{-1.0, 1.0} {}
    double operator()() { return _dist(_rng); }
    std::mt19937 _rng;
    std::uniform_real_distribution<> _dist;
  };

  void fill_weights(size_t num, bool use_random) {
    auto gen = UniformRandomGenerator();
    for (auto i = 0; i < num; ++i) {
      auto val = 1.0;
      if (use_random) {
        val = gen();
      }
      _w.emplace_back(make_shared<Value>(val));
    }
  }

  ValuePtr operator()(vector<ValuePtr> x) {
    auto wx = x[0] * _w[0];
    for (auto i = 1; i < x.size(); ++i) {
      wx = wx + (x[i] * _w[i]);
    }
    auto act = wx + _b;
    if (_non_linear) {
      return act->relu();
    }
    return act;
  }

  friend ostream& operator<<(ostream& os, const Neuron& val) {
    auto act = "Linear";
    if (val._non_linear) { act = "ReLU"; }
    os << act << "Neuron(" << val._w.size() << ")";
    return os;
  }

  vector<ValuePtr> parameters() {
    auto whole = _w;
    whole.push_back(_b);
    return whole;
  }

  vector<ValuePtr> _w;
  ValuePtr _b;
  bool _non_linear;
};

struct Layer : public Module {
  Layer(size_t in_nr, size_t out_nr, bool non_linear = true,
        bool is_test = false)
      : _in_nr{in_nr}, _out_nr{out_nr}, _neurons{} {
    fill_neurons(non_linear, is_test);
  }
  ~Layer() {}

  void fill_neurons(bool non_linear, bool is_test) {
    for (auto i = 0; i < _out_nr; ++i) {
      _neurons.emplace_back(_in_nr, non_linear, is_test);
    }
  }

  vector<ValuePtr> operator()(vector<ValuePtr> x) {
    auto out = vector<ValuePtr>{};
    for (auto neuron : _neurons) {
      out.emplace_back(neuron(x));
    }
    return out;
  }

  friend ostream& operator<<(ostream& os, const Layer& layer) {
    std::string str = "Layer of[";
    for (auto& n: layer._neurons) {
      std::stringstream ss;
      ss << n;
      str += ss.str() + ", ";
    }
    str = str.substr(0, str.size() - 2);
    str += "]";
    os << str;
    return os;
  }


  vector<ValuePtr> parameters() {
    vector<ValuePtr> whole;
    for (auto neuron: _neurons) {
      for (auto param: neuron.parameters()) {
        whole.push_back(param);
      }
    }
    return whole;
  }

  size_t _in_nr;
  size_t _out_nr;
  vector<Neuron> _neurons;
};

struct MLP : public Module {
  MLP(size_t in_nr, std::vector<size_t> outs_nr, bool is_test = false) : _layers() {
    vector<size_t> sz(outs_nr.begin(), outs_nr.end());
    sz.insert(sz.begin(), in_nr);
    for (auto i = 0; i < outs_nr.size(); ++i) {
      _layers.emplace_back(sz[i], sz[i + 1], i != (outs_nr.size() - 1),
                           is_test);
    }
  }
  MLP(size_t in_nr, std::initializer_list<size_t> outs_nr, bool is_test = false)
    : MLP(in_nr, std::vector<size_t>{outs_nr}, is_test) {
  }
  ~MLP() {}

  vector<ValuePtr> operator()(vector<ValuePtr> x) {
    for (auto layer : _layers) {
      x = layer(x);
    }
    return x;
  }

  friend ostream& operator<<(ostream& os, const MLP& mlp) {
    std::string str = "MLP of[";
    for (auto& layer: mlp._layers) {
      std::stringstream ss;
      ss << layer;
      str += ss.str() + ", ";
    }
    str = str.substr(0, str.size() - 2);
    str += "]";
    os << str;
    return os;
  }

  vector<ValuePtr> parameters() {
    vector<ValuePtr> whole;
    for (auto layer: _layers) {
      for (auto param: layer.parameters()) {
        whole.push_back(param);
      }
    }
    return whole;
  }

  vector<Layer> _layers;
};

}  // namespace ugrad

#endif  // __UGRAD_NN_HPP__