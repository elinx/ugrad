#include "engine.hpp"
#include <vector>
#include "gtest/gtest.h"

using ugrad::Value;
using std::vector;

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
  EXPECT_EQ(c.children()[0].get().data(), a.data());
  EXPECT_EQ(c.children()[1].get().data(), b.data());
}

TEST(ValueTest, SelfAddChildren) {
  auto a = Value{-1.0};
  a += -2;
  ASSERT_EQ(a.children().size(), 0);
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

  ASSERT_FALSE(c.children().empty());
  ASSERT_EQ(c.children().size(), 2);
  ASSERT_TRUE(c.children()[0].get().children().empty());
  ASSERT_TRUE(c.children()[1].get().children().empty());

  EXPECT_EQ(c.children()[0].get().data(), -1.0);
  EXPECT_EQ(c.children()[1].get().data(), 1.0);
  // EXPECT_EQ(c.children()[0].get().children()[0].get().data(), a.data());
  // EXPECT_EQ(c.children()[0].get().children()[1].get().data(), b.data());
}

TEST(ValueTest, TopoSortEasy) {
  auto a = Value{-1.0};
  auto b = Value{1.0};
  auto c = a + b;
  auto topo_order = c.build_topo();

  ASSERT_FALSE(topo_order.empty());
  ASSERT_EQ(topo_order.size(), 3);
  EXPECT_EQ(topo_order[0].get().data(), 0);   // c
  EXPECT_EQ(topo_order[1].get().data(), 1);   // b
  EXPECT_EQ(topo_order[2].get().data(), -1);  // a

  topo_order = c.build_topo();

  ASSERT_FALSE(topo_order.empty());
  ASSERT_EQ(topo_order.size(), 3);
  EXPECT_EQ(topo_order[0].get().data(), 0);  // c
  EXPECT_EQ(topo_order[1].get().data(), 1);  // b
  EXPECT_EQ(topo_order[2].get().data(), -1); // a
}

TEST(ValueTest, TopoSortEasy1) {
  auto a = Value{-1.0};
  auto b = Value{1.0};
  auto c = a + b;  // c'
  printf("c addr: %p\n", &c);
  c += b;
  printf("c addr: %p\n", &c);

  auto topo_order = c.build_topo();

  ASSERT_FALSE(topo_order.empty());
  ASSERT_EQ(topo_order.size(), 5);

  EXPECT_EQ(topo_order[0].get().data(), 1);  // c
  EXPECT_EQ(topo_order[1].get().data(), 1);  // b
  EXPECT_EQ(topo_order[2].get().data(), 0);  // c'
  EXPECT_EQ(topo_order[3].get().data(), 1);  // b
  EXPECT_EQ(topo_order[4].get().data(), -1); // a
}

TEST(ValueTest, GradSimple1) {
  auto a = Value{-99.0f};
  a.backward();
  EXPECT_EQ(a.data(), -99.0);
  EXPECT_EQ(a.grad(), 1.0);
}

TEST(ValueTest, GradSimple2) {
  auto a = Value{-4.0f};
  auto b = Value{2.0f};
  auto c = a + b;
  c.backward();
  EXPECT_EQ(a.grad(), 1.0);
  EXPECT_EQ(b.grad(), 1.0);
  EXPECT_EQ(c.grad(), 1.0);
}