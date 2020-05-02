#include "gtest/gtest.h"
#include "engine.hpp"

using ugrad::Value;

TEST(ValueTest, Add) {
  auto a = Value{-1.0};
  auto b = Value{1.0};
  auto c = a + b;
  EXPECT_EQ(0, c.data());
}

TEST(ValueTest, SelfAdd) {
  auto a = Value{-1.0};
  a += -2;
  EXPECT_EQ(-3, a.data());
}

TEST(ValueTest, Negtivate) {
  auto a = Value{-1.0};
  a = -a;
  EXPECT_EQ(1, a.data());
}

TEST(ValueTest, AddChildren) {
  auto a = Value{-1.0};
  auto b = Value{1.0};
  auto c = a + b;
  EXPECT_EQ(c.children()[0].data(), a.data());
  EXPECT_EQ(c.children()[1].data(), b.data());
}

TEST(ValueTest, SelfAddChildren) {
  auto a = Value{-1.0};
  a += -2;
  EXPECT_EQ(a.children()[0].data(), -1.0);
  EXPECT_EQ(a.children()[1].data(), -2.0);
}

TEST(ValueTest, SelfAddChildrenAddr) {
  auto a = Value{-1.0};
  auto pa = &a;
  a += -2;
  EXPECT_EQ(&a, pa);
}

TEST(ValueTest, SelfAddNest) {
  auto a = Value{-1.0};
  auto b = Value{1.0};
  auto c = a + b;
  c += 2.0;
  EXPECT_EQ(c.children()[0].data(), 0.0);
  EXPECT_EQ(c.children()[1].data(), 2.0);
  EXPECT_EQ(c.children()[0].children()[0].data(), a.data());
  EXPECT_EQ(c.children()[0].children()[1].data(), b.data());
}