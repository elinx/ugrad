#include <fmt/core.h>
#include <fmt/ostream.h>

#include <fstream>
#include <iostream>
#include <tuple>
#include <ugrad/engine.hpp>
#include <ugrad/nn.hpp>

using std::ifstream;
using std::tuple;

using namespace ugrad;

static vector<vector<ValuePtr>> read_dataset_x(const char* xfile) {
  vector<vector<ValuePtr>> X;
  ifstream xstr(xfile);
  if (!xstr.is_open()) {
    fmt::print("failed to open {} file\n", xfile);
    return {};
  }

  double x1, x2;
  while (xstr >> x1 >> x2) {
    X.emplace_back(
        vector<ValuePtr>{make_shared<Value>(x1), make_shared<Value>(x2)});
  }
  return X;
}

static vector<vector<ValuePtr>> read_dataset_y(const char* yfile) {
  vector<vector<ValuePtr>> y;
  ifstream ystr(yfile);
  if (!ystr.is_open()) {
    fmt::print("failed to open {} file\n", yfile);
    return {};
  }

  double y1;
  while (ystr >> y1) {
    y.emplace_back(vector<ValuePtr>{make_shared<Value>(y1)});
  }
  return y;
}

static tuple<vector<vector<ValuePtr>>, vector<vector<ValuePtr>>> read_dataset(
    const char* xfile, const char* yfile) {
  auto X = read_dataset_x(xfile);
  if (X.empty()) {
    return {};
  }
  auto y = read_dataset_y(yfile);
  if (y.empty()) {
    return {};
  }
  return std::make_tuple(X, y);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    fmt::print("Usage: mlp_example X.txt y.txt\n");
    return -1;
  }

  auto [X, y] = read_dataset(argv[1], argv[2]);
  fmt::print("read dataset finished, size of X: {}, size of y: {}\n", X.size(),
             y.size());

  auto model = MLP(2, {16, 16, 1});
  fmt::print("model: {}\n", model);

  return 0;
}