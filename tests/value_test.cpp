#include <memory>
#include <vector>

#include <ugrad/engine.hpp>
#include <gtest/gtest.h>

using std::make_shared;
using std::vector;
using ugrad::Value;

TEST(ValueTest, Add) {
  auto a = make_shared<Value>(-1.0);
  auto b = make_shared<Value>(1.0);
  auto c = a + b;
  EXPECT_EQ(0, c->data());
}

TEST(ValueTest, SelfAdd) {
  auto a = make_shared<Value>(-1.0);
  auto const2 = make_shared<Value>(-2.0);
  a = a + const2;
  EXPECT_EQ(-3, a->data());
}

TEST(ValueTest, Negtivate) {
  auto a = make_shared<Value>(-1.0);
  a = -a;
  EXPECT_EQ(1, a->data());
}

TEST(ValueTest, AddChildren) {
  auto a = make_shared<Value>(-1.0);
  auto b = make_shared<Value>(1.0);
  auto c = a + b;
  EXPECT_EQ(c->data(), 0.0);
  EXPECT_EQ(c->children()[0]->data(), a->data());
  EXPECT_EQ(c->children()[1]->data(), b->data());
}

TEST(ValueTest, SelfAddChildren) {
  auto a = make_shared<Value>(-1.0);
  auto const2 = make_shared<Value>(-2.0);
  a = a + const2;
  ASSERT_EQ(a->children().size(), 2);
}

TEST(ValueTest, SelfAddChildrenAddr) {
  auto a = make_shared<Value>(-1.0);
  auto const2 = make_shared<Value>(-2.0);
  auto pa = a.get();
  a = a + const2;
  EXPECT_NE(a.get(), pa);
}

TEST(ValueTest, SelfAddNest) {
  auto a = make_shared<Value>(-1.0);
  auto b = make_shared<Value>(1.0);
  auto const2 = make_shared<Value>(2.0);

  auto c = a + b;
  c = c + const2;

  ASSERT_FALSE(c->children().empty());
  ASSERT_EQ(c->children().size(), 2);
  ASSERT_FALSE(c->children()[0]->children().empty());
  ASSERT_TRUE(c->children()[1]->children().empty());

  EXPECT_EQ(c->children()[0]->data(), 0.0);
  EXPECT_EQ(c->children()[1]->data(), 2.0);
  EXPECT_EQ(c->children()[0]->children()[0]->data(), a->data());
  EXPECT_EQ(c->children()[0]->children()[1]->data(), b->data());
}

TEST(ValueTest, TopoSortEasy) {
  auto a = make_shared<Value>(-1.0);
  auto b = make_shared<Value>(1.0);
  auto c = a + b;
  auto topo_order = c->build_topo();

  ASSERT_FALSE(topo_order.empty());
  ASSERT_EQ(topo_order.size(), 3);
  EXPECT_EQ(topo_order[0]->data(), 0);   // c
  EXPECT_EQ(topo_order[1]->data(), 1);   // b
  EXPECT_EQ(topo_order[2]->data(), -1);  // a

  topo_order = c->build_topo();

  ASSERT_FALSE(topo_order.empty());
  ASSERT_EQ(topo_order.size(), 3);
  EXPECT_EQ(topo_order[0]->data(), 0);   // c
  EXPECT_EQ(topo_order[1]->data(), 1);   // b
  EXPECT_EQ(topo_order[2]->data(), -1);  // a
}

TEST(ValueTest, TopoSortEasy1) {
  auto a = make_shared<Value>(-1.0);
  auto b = make_shared<Value>(1.0);
  auto c = a + b;  // c'

  c = c + b;  // c

  auto topo_order = c->build_topo();

  ASSERT_FALSE(topo_order.empty());
  ASSERT_EQ(topo_order.size(), 4);

  EXPECT_EQ(topo_order[0]->data(), 1);   // c
  EXPECT_EQ(topo_order[1]->data(), 0);   // c'
  EXPECT_EQ(topo_order[2]->data(), 1);   // b
  EXPECT_EQ(topo_order[3]->data(), -1);  // a
}

TEST(GradTest, Single) {
  auto a = make_shared<Value>(-99.0f);
  a->backward();
  EXPECT_EQ(a->data(), -99.0);
  EXPECT_EQ(a->grad(), 1.0);
}

TEST(GradTest, Add1) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a + b;
  c->backward();
  EXPECT_EQ(a->grad(), 1.0);
  EXPECT_EQ(b->grad(), 1.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, Mul1) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a * b;
  c->backward();
  EXPECT_EQ(a->grad(), 2.0);
  EXPECT_EQ(b->grad(), -4.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, Power) {
  auto a = make_shared<Value>(-4.0f);
  auto c = a * a * a;
  c->backward();
  EXPECT_EQ(a->grad(), 48);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, UseMultipleTimes) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a * b;
  c = c * b;  // c = a * (b*b)
  c->backward();
  EXPECT_EQ(a->grad(), 4.0);
  EXPECT_EQ(b->grad(), -16.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, ReluPos) {
  auto a = make_shared<Value>(4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a * b;
  c = c->relu() * b;  // c = relu(a*b) * b
  c->backward();
  EXPECT_EQ(a->grad(), 4.0);
  EXPECT_EQ(b->grad(), 16.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, ReluNeg) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a * b;
  c = c->relu() * b;  // c = relu(a*b) * b
  c->backward();
  EXPECT_EQ(a->grad(), 0.0);
  EXPECT_EQ(b->grad(), 0.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, PowTest) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a->pow(b);
  c->backward();
  EXPECT_EQ(a->grad(), -8.0);
  EXPECT_EQ(b->grad(), 0.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, PowTestNeg) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(-1.0f);
  auto c = a->pow(b);
  c->backward();
  EXPECT_EQ(a->grad(), -1.0f / 16);
  EXPECT_EQ(b->grad(), 0.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, DivTest) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(-1.0f);
  auto c = a / b;
  c->backward();
  EXPECT_EQ(a->grad(), -1.0f);
  EXPECT_EQ(b->grad(), 4.0);
  EXPECT_EQ(c->grad(), 1.0);
}

TEST(GradTest, SanityCheck) {
  auto x = make_shared<Value>(-4.0f);
  auto z = make_shared<Value>(2.0f) * x + make_shared<Value>(2.0f) + x;
  auto q = z->relu() + z * x;
  auto h = (z * z)->relu();
  auto y = h + q + q * x;
  y->backward();
  EXPECT_EQ(y->data(), -20);
  EXPECT_EQ(x->grad(), 46);
}

TEST(GradTest, MoreOps1) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a + b;
  auto d = a * b + b * b * b;
  c = c + c + make_shared<Value>(1.0f);
  c = c + make_shared<Value>(1.0f) + c + (-a);
  d = d + d * make_shared<Value>(2.0f) + (b + a)->relu();
  d->backward();
  EXPECT_FLOAT_EQ(d->data(), 0.0);
  EXPECT_FLOAT_EQ(a->grad(), 6.0);
  EXPECT_FLOAT_EQ(b->grad(), 24.0);
}

TEST(GradTest, MoreOps2) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a + b;
  auto d = a * b + b * b * b;
  c = c + c + make_shared<Value>(1.0f);
  c = c + make_shared<Value>(1.0f) + c + (-a);
  d = d + d * make_shared<Value>(2.0f) + (b + a)->relu();
  d = d + make_shared<Value>(3.0f) * d + (b - a)->relu();
  d->backward();
  EXPECT_FLOAT_EQ(d->data(), 6.0);
  EXPECT_FLOAT_EQ(a->grad(), 23.0);
  EXPECT_FLOAT_EQ(b->grad(), 97.0);
}

TEST(GradTest, MoreOps) {
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a + b;
  auto d = a * b + b * b * b;
  c = c + c + make_shared<Value>(1.0f);
  c = c + make_shared<Value>(1.0f) + c + (-a);
  d = d + d * make_shared<Value>(2.0f) + (b + a)->relu();
  d = d + make_shared<Value>(3.0f) * d + (b - a)->relu();
  auto e = c - d;
  auto f = e * e;
  auto g = f / make_shared<Value>(2.0f);
  g = g + make_shared<Value>(10.0f) / f;
  g->backward();
  EXPECT_FLOAT_EQ(g->data(), 24.704082);
  EXPECT_FLOAT_EQ(a->grad(), 138.833819);
  EXPECT_FLOAT_EQ(b->grad(), 645.577259);
}
