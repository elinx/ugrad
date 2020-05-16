#include <iostream>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <ugrad/engine.hpp>

using namespace ugrad;

int main()
{
  auto a = make_shared<Value>(-4.0f);
  auto b = make_shared<Value>(2.0f);
  auto c = a + b;
  auto d = a * b + b * b * b;
  c = c + c + 1;
  c = c + 1 + c + (-a);
  d = d + d * 2 + (b + a)->relu();
  d = d + 3 * d + (b - a)->relu();
  auto e = c - d;
  auto f = e * e;
  auto g = f / 2.0;
  g = g + 10.0 / f;
  fmt::print("g: {}\n", *g); // prints 24.7041, the outcome of this forward pass
  g->backward();
  fmt::print("a: {}\n", *a); // prints 138.8338, i.e. the numerical value of dg/da
  fmt::print("b: {}\n", *b); // prints 645.5773, i.e. the numerical value of dg/db
  return 0;
}