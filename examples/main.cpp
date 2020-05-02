#include <iostream>
#include "fmt/core.h"
#include "fmt/ostream.h"
#include "engine.hpp"

using namespace ugrad;

int main()
{
  auto kk = Value{-99.0f};
  fmt::print("-kk: {}\n", -kk);

  auto a = Value{-4.0f};
  auto b = Value{2.0f};
  auto c = a + b;
  auto d = a * b + b * b * b;
  c += c + 1;
  c += 1 + c + (-a);
  d += d * 2 + (b + a).relu();
  d += 3 * d + (b - a).relu();
  auto e = c - d;
  auto f = e * e;
  auto g = f / 2.0;
  g += 10.0 / f;
  fmt::print("g: {}\n", g); // prints 24.7041, the outcome of this forward pass
  // g.backward()
  // print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
  // print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
  return 0;
}