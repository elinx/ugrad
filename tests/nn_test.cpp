#include <memory>
#include <vector>

#include <ugrad/engine.hpp>
#include <ugrad/nn.hpp>
#include <gtest/gtest.h>

using std::make_shared;
using std::vector;
using ugrad::Value;
using ugrad::ValuePtr;
using ugrad::Module;
using ugrad::Neuron;
using ugrad::Layer;
using ugrad::MLP;

constexpr bool static is_test = true;
constexpr bool static relu_act = true;
constexpr bool static no_act = false;

TEST(NeuronTest, ReluAct) {
  auto n = Neuron(2, relu_act, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(-2.0)
  };
  auto y = n(x);
  EXPECT_EQ(0, y->data());
}

TEST(NeuronTest, NoAct) {
  auto n = Neuron(2, no_act, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(-2.0)
  };
  auto y = n(x);
  EXPECT_EQ(-1.0, y->data());
}

TEST(LayerTest, ReluAct) {
  const size_t in_nr = 2;
  const size_t out_nr = 1;
  auto n = Layer(in_nr, out_nr, relu_act, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(-2.0)
  };
  auto y = n(x);
  ASSERT_EQ(out_nr, y.size());
  EXPECT_EQ(0, y[0]->data());
}

TEST(LayerTest, ReluAct2) {
  const size_t in_nr = 2;
  const size_t out_nr = 3;
  auto n = Layer(in_nr, out_nr, relu_act, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(-2.0)
  };
  auto y = n(x);
  ASSERT_EQ(out_nr, y.size());
  EXPECT_EQ(0, y[0]->data());
  EXPECT_EQ(0, y[1]->data());
  EXPECT_EQ(0, y[2]->data());
}

TEST(LayerTest, NoAct) {
  const size_t in_nr = 2;
  const size_t out_nr = 1;
  auto n = Layer(in_nr, out_nr, no_act, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(-2.0)
  };
  auto y = n(x);
  ASSERT_EQ(out_nr, y.size());
  EXPECT_EQ(-1.0, y[0]->data());
}

TEST(LayerTest, NoAct2) {
  const size_t in_nr = 2;
  const size_t out_nr = 3;
  auto n = Layer(in_nr, out_nr, no_act, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(-2.0)
  };
  auto y = n(x);
  ASSERT_EQ(out_nr, y.size());
  EXPECT_EQ(-1, y[0]->data());
  EXPECT_EQ(-1, y[1]->data());
  EXPECT_EQ(-1, y[2]->data());
}

TEST(MLPTest, ReluActZero) {
  const size_t in_nr = 2;
  auto outs_nr = {4llu, 4llu, 1llu};
  auto n = MLP(in_nr, outs_nr, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(-2.0)
  };
  auto y = n(x);
  ASSERT_EQ(*std::rbegin(outs_nr), y.size());
  EXPECT_EQ(0.0, y[0]->data());
}

TEST(MLPTest, ReluActPos) {
  const size_t in_nr = 2;
  auto outs_nr = {4llu, 4llu, 1llu};
  auto n = MLP(in_nr, outs_nr, is_test);
  auto x = vector<ValuePtr>{
    make_shared<Value>(1.0),
    make_shared<Value>(2.0)
  };
  auto y = n(x);
  ASSERT_EQ(*std::rbegin(outs_nr), y.size());
  EXPECT_EQ(48.0, y[0]->data());
}