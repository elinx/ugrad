# ugrad
A C++ implementation of the scalar-valued autograd engine [micrograd](https://github.com/karpathy/micrograd)

## Example Usage
```c++
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
```

## MLP Example

```shell
read dataset finished, size of X: 100, size of y: 100
model: MLP of[Layer of[ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2)], Layer of[ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16), ReLUNeuron(16)], Layer of[LinearNeuron(16)]]
number of parameters: 337
epoch 0 loss 1.2725152912873847, accuracy 50.00%, lr: 1.0000
epoch 1 loss 1.9048338557526878, accuracy 66.00%, lr: 0.9910
epoch 2 loss 0.7485335053477321, accuracy 70.00%, lr: 0.9820
epoch 3 loss 0.92274670070065, accuracy 82.00%, lr: 0.9730
epoch 4 loss 0.3142488652855841, accuracy 85.00%, lr: 0.9640
epoch 5 loss 0.2910938000612888, accuracy 87.00%, lr: 0.9550
epoch 6 loss 0.28777670574329695, accuracy 86.00%, lr: 0.9460
epoch 7 loss 0.2890573596107362, accuracy 92.00%, lr: 0.9370
epoch 8 loss 0.23966708205580817, accuracy 92.00%, lr: 0.9280
epoch 9 loss 0.2364786200105814, accuracy 94.00%, lr: 0.9190
epoch 10 loss 0.2254051355225215, accuracy 92.00%, lr: 0.9100
epoch 11 loss 0.22035147226006968, accuracy 94.00%, lr: 0.9010
epoch 12 loss 0.19991848684099722, accuracy 92.00%, lr: 0.8920
epoch 13 loss 0.20509400679406434, accuracy 95.00%, lr: 0.8830
epoch 14 loss 0.18829085067792212, accuracy 92.00%, lr: 0.8740
epoch 15 loss 0.1974543475293526, accuracy 93.00%, lr: 0.8650
epoch 16 loss 0.17242809209578425, accuracy 92.00%, lr: 0.8560
epoch 17 loss 0.16186244456015614, accuracy 95.00%, lr: 0.8470
epoch 18 loss 0.16104795341728673, accuracy 92.00%, lr: 0.8380
epoch 19 loss 0.17195409065398698, accuracy 95.00%, lr: 0.8290
epoch 20 loss 0.18654272523260412, accuracy 92.00%, lr: 0.8200
epoch 21 loss 0.18873687045318108, accuracy 95.00%, lr: 0.8110
epoch 22 loss 0.11898904791539117, accuracy 95.00%, lr: 0.8020
epoch 23 loss 0.10374206751321746, accuracy 96.00%, lr: 0.7930
epoch 24 loss 0.11133809018509823, accuracy 96.00%, lr: 0.7840
epoch 25 loss 0.0869769948969216, accuracy 97.00%, lr: 0.7750
epoch 26 loss 0.09244877197388525, accuracy 97.00%, lr: 0.7660
epoch 27 loss 0.10944969528256565, accuracy 96.00%, lr: 0.7570
epoch 28 loss 0.14940032092337396, accuracy 95.00%, lr: 0.7480
epoch 29 loss 0.1128000518442734, accuracy 94.00%, lr: 0.7390
epoch 30 loss 0.07793686300183608, accuracy 98.00%, lr: 0.7300
epoch 31 loss 0.09546874752129621, accuracy 97.00%, lr: 0.7210
epoch 32 loss 0.06806739869763159, accuracy 97.00%, lr: 0.7120
epoch 33 loss 0.05176941565511138, accuracy 98.00%, lr: 0.7030
epoch 34 loss 0.05006231679840696, accuracy 99.00%, lr: 0.6940
epoch 35 loss 0.05062781192019704, accuracy 98.00%, lr: 0.6850
epoch 36 loss 0.07844651885993426, accuracy 98.00%, lr: 0.6760
epoch 37 loss 0.08421588994620477, accuracy 97.00%, lr: 0.6670
epoch 38 loss 0.04321012633792009, accuracy 98.00%, lr: 0.6580
epoch 39 loss 0.03736974183828634, accuracy 100.00%, lr: 0.6490
epoch 40 loss 0.04322283539319244, accuracy 98.00%, lr: 0.6400
epoch 41 loss 0.03445071314616174, accuracy 100.00%, lr: 0.6310
epoch 42 loss 0.03964836995260649, accuracy 99.00%, lr: 0.6220
epoch 43 loss 0.030669220672464005, accuracy 100.00%, lr: 0.6130
epoch 44 loss 0.03181888995654795, accuracy 100.00%, lr: 0.6040
epoch 45 loss 0.02512036603837519, accuracy 100.00%, lr: 0.5950
epoch 46 loss 0.032500953145045565, accuracy 100.00%, lr: 0.5860
epoch 47 loss 0.022972301417005618, accuracy 100.00%, lr: 0.5770
epoch 48 loss 0.03838832612171553, accuracy 99.00%, lr: 0.5680
epoch 49 loss 0.017589800872646832, accuracy 100.00%, lr: 0.5590
epoch 50 loss 0.024005346205426367, accuracy 100.00%, lr: 0.5500
epoch 51 loss 0.050508980108954264, accuracy 98.00%, lr: 0.5410
```
