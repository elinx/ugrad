#include "gtest/gtest.h"
#include "engine.hpp"

using ugrad::Value;

TEST(ValueTest, Add) {
  auto a = Value{-1.0};
  auto b = Value{1.0};
  auto c = a + b;
  EXPECT_EQ(0, c.data());
}