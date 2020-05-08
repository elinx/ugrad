#include <fmt/core.h>
#include <fmt/ostream.h>

#include <fstream>
#include <iostream>
#include <tuple>
#include <ugrad/engine.hpp>
#include <ugrad/nn.hpp>
#include <algorithm>
#include <numeric>

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

static vector<vector<ValuePtr>> forward(MLP& model,
                                        const vector<vector<ValuePtr>>& inputs) {
  vector<vector<ValuePtr>> scores;
  for (auto input : inputs) {
    scores.emplace_back(model(input));
  }
  return scores;
}

static tuple<ValuePtr, double> loss(const vector<vector<ValuePtr>>& scores,
                                    const vector<vector<ValuePtr>> y,
                                    const vector<ValuePtr>& parameters) {
  vector<ValuePtr> losses;
  for (auto i = 0; i < y.size(); ++i) {
    // fmt::print("scores[{}]: {:.6f}, y[{}]: {}\n", i, scores[i][0]->data(), i, y[i][0]->data());
    losses.emplace_back(
        (make_shared<Value>(1.0) + (-y[i][0]) * scores[i][0])->relu());
  }
  // svm "max-margin" loss
  auto data_loss = std::accumulate(losses.begin(), losses.end(), make_shared<Value>(0.0));
  data_loss = data_loss / make_shared<Value>(losses.size());

  // L2 regularization
  auto alpha = make_shared<Value>(1e-4);
  auto square_sum = std::inner_product(parameters.begin(), parameters.end(),
    parameters.begin(), make_shared<Value>(0.0));
  auto reg_loss = alpha * square_sum;
  auto total_loss = data_loss + reg_loss;

  // get accuracy
  double accuracy = 0.0;
  for (auto i = 0; i < y.size(); ++i) {
    accuracy += (scores[i][0]->data() > 0) == (y[i][0]->data() > 0);
  }
  accuracy /= y.size();
  return std::make_tuple(total_loss, accuracy);
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
  fmt::print("number of parameters: {}\n", model.parameters().size());

  const size_t epochs = 100;
  for (auto epoch = 0; epoch < epochs; ++epoch) {
    auto scores = forward(model, X);
    auto [total_loss, acc] = loss(scores, y, model.parameters());
    // fmt::print("total loss: {}, accuracy: {}\n", *total_loss, acc);

    model.zero_grad();
    total_loss->backward();

    double learning_rate = 1.0 - 0.9 * epoch / 100;
    learning_rate = std::max(learning_rate, 0.001);
    for (auto p : model.parameters()) {
      // fmt::print("++ p->data: {:.6f}, p->grad: {:.6f}\n", p->data(), p->grad());
      p->_data -= learning_rate * p->grad();
      // fmt::print("-- p->data: {:.6f}, p->grad: {:.6f}\n", p->data(), p->grad());
    }

    fmt::print("epoch {} loss {}, accuracy {:.2f}%, lr: {:.4f}\n", epoch, total_loss->data(),
               acc * 100, learning_rate);
  }

  return 0;
}